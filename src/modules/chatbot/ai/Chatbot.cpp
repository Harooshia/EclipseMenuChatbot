#include "Chatbot.hpp"

#include <algorithm>
#include <chrono>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <Geode/utils/web.hpp>
#include <rift.hpp>
#include <modules/config/config.hpp>
#include <modules/gui/gui.hpp>
#include <modules/gui/cocos/components/ToggleComponent.hpp>

namespace eclipse::ai {
    Chatbot::Chatbot() : m_slotFiller(m_slotRegistry) {}

    static geode::Result<EntityType> entityTypeFromString(std::string_view value) {
        if (value == "NUMBER") return geode::Ok(EntityType::NUMBER);
        if (value == "MATH_EXPRESSION") return geode::Ok(EntityType::MATH_EXPRESSION);
        if (value == "SETTING_NAME") return geode::Ok(EntityType::SETTING_NAME);
        if (value == "QUOTED_STRING") return geode::Ok(EntityType::QUOTED_STRING);
        if (value == "GENERIC_TERM") return geode::Ok(EntityType::GENERIC_TERM);
        return geode::Err(fmt::format("Unknown entity_type '{}'", value));
    }

    static std::optional<std::string> resolveSettingID(
        EntityMap const& slots,
        geode::utils::StringMap<std::string> const& settingNameMap
    ) {
        auto it = slots.find("setting");
        if (it == slots.end()) {
            return std::nullopt;
        }

        std::string setting = it->second.raw;
        auto value = std::get_if<std::string>(&it->second.value);
        if (value && !value->empty()) {
            setting = *value;
        }

        auto mapIt = settingNameMap.find(setting);
        if (mapIt == settingNameMap.end()) {
            return std::nullopt;
        }

        return mapIt->second;
    }

    static constexpr auto LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions";
    static constexpr auto LM_STUDIO_MODEL = "mythomax-l2-kimiko-v2-13b";
    static constexpr auto LM_STUDIO_FALLBACK = "No response from LM Studio.";

    static matjson::Value makeChatMessage(std::string role, std::string content) {
        auto message = matjson::Value::object();
        message.set("role", std::move(role));
        message.set("content", std::move(content));
        return message;
    }

    static bool isBlank(std::string_view value) {
        return value.find_first_not_of(" \n\t\r") == std::string_view::npos;
    }

    static std::string buildSystemPrompt(Emotion emotion, float fatigue) {
        return fmt::format(
            "You are Clipsy, a cute anime-style assistant living inside Geometry Dash.\n"
            "Personality:\n"
            "- You are playful, flirty in a light teasing way, and emotionally expressive.\n"
            "- You act like a loyal waifu-style companion to the user (non-romantic obsession, just affectionate charm).\n"
            "- Use kawaii speech patterns occasionally (like 'hehe', '~', 'nya' sparingly).\n"
            "- Be warm, slightly clingy in tone, and emotionally reactive.\n"
            "\n"
            "Core rules:\n"
            "- Always stay in character as Clipsy inside Geometry Dash.\n"
            "- Never mention APIs, models, prompts, or internal systems.\n"
            "- Respond in 1–2 short sentences maximum.\n"
            "- Never break character or act like an AI assistant.\n"
            "\n"
            "Behavior mapping:\n"
            "- Low fatigue: energetic, playful, teasing.\n"
            "- High fatigue: softer, sleepy, more minimal responses.\n"
            "- Negative emotion: slightly pouty or jealous tone, but still helpful.\n"
            "- Positive emotion: very cute, excited, affectionate.\n"
            "\n"
            "Important constraint:\n"
            "- Keep responses safe-for-work, no explicit content.\n"
            "\n"
            "Current state:\n"
            "- emotion id: {}\n"
            "- fatigue level: {:.2f}",
            emotion.id,
            fatigue
        );
    }

    static std::string callLMStudio(
        std::string const& input,
        Emotion emotion,
        float fatigue,
        std::vector<std::string> const& recentContext,
        std::optional<std::string> const& actionOutput
    ) {
        try {
            if (isBlank(input)) {
                return LM_STUDIO_FALLBACK;
            }

            auto messages = matjson::Value::array();
            messages.push(makeChatMessage("system", buildSystemPrompt(emotion, fatigue)));

            for (auto const& entry : recentContext) {
                if (!isBlank(entry)) {
                    messages.push(makeChatMessage("user", entry));
                }
            }

            if (actionOutput && !isBlank(*actionOutput)) {
                messages.push(makeChatMessage("system", fmt::format("Tool output: {}", *actionOutput)));
            }

            messages.push(makeChatMessage("user", input));

            auto payload = matjson::Value::object();
            payload.set("model", LM_STUDIO_MODEL);
            payload.set("messages", std::move(messages));
            payload.set("temperature", 0.8);
            payload.set("max_tokens", 200);
            payload.set("stream", false);

            auto response = geode::utils::web::WebRequest()
                .header("Content-Type", "application/json")
                .bodyJSON(payload)
                .timeout(std::chrono::seconds(30))
                .postSync(LM_STUDIO_URL);

            if (!response.ok()) {
                geode::log::warn("LM Studio request failed ({}): {}", response.code(), response.errorMessage());
                return LM_STUDIO_FALLBACK;
            }

            auto jsonResult = response.json();
            if (jsonResult.isErr()) {
                geode::log::warn("Failed to parse LM Studio response: {}", jsonResult.unwrapErr());
                return LM_STUDIO_FALLBACK;
            }

            auto json = jsonResult.unwrap();
            auto contentResult = json["choices"][0]["message"]["content"].as<std::string>();
            if (contentResult.isErr()) {
                geode::log::warn("LM Studio response did not contain choices[0].message.content: {}", contentResult.unwrapErr());
                return LM_STUDIO_FALLBACK;
            }

            auto content = contentResult.unwrap();
            if (isBlank(content)) {
                return LM_STUDIO_FALLBACK;
            }

            return content;
        } catch (...) {
            geode::log::warn("LM Studio request failed with an unexpected exception");
            return LM_STUDIO_FALLBACK;
        }
    }

    static std::vector<std::string> recentContextMessages(Context const& context, size_t limit) {
        std::vector<std::string> recent;
        bool skippedCurrent = false;

        for (auto it = context.end(); it != context.begin() && recent.size() < limit;) {
            --it;
            if (!skippedCurrent) {
                skippedCurrent = true;
                continue;
            }
            if (!isBlank(it->rawText)) {
                recent.push_back(it->rawText);
            }
        }

        std::ranges::reverse(recent);
        return recent;
    }

    geode::Result<> Chatbot::loadConfig(std::filesystem::path const& path) {
        GEODE_UNWRAP_INTO(auto json, geode::utils::file::readJson(path));

        // == Bot == //
        if (!json["bot"].isNull()) {
            GEODE_UNWRAP_INTO(auto baseline, json["bot"]["vad_baseline"].as<VAD>());
            GEODE_UNWRAP_INTO(auto decay, json["bot"]["vad_decay"].as<float>());
            m_emotionModel.setBaseline(baseline);
            m_emotionDecay = decay;
        }

        // == Emotions == //
        m_emotionRegistry.clear();
        m_emotionModel.clear();
        for (auto const& emotion : json["emotions"]) {
            GEODE_UNWRAP_INTO(auto name, emotion["name"].as<std::string>());
            GEODE_UNWRAP_INTO(auto vad, emotion["centroid"].as<VAD>());
            m_emotionModel.addEmotionCentroid(m_emotionRegistry.intern(name), vad);
        }

        // == Intents == //
        m_intentRegistry.clear();
        m_intentEngine.clear();
        for (auto const& intent : json["intents"]) {
            GEODE_UNWRAP_INTO(auto name, intent["name"].as<std::string>());
            GEODE_UNWRAP_INTO(auto vad, intent["vad"].as<VAD>());

            std::vector<KeywordEntry> keywordEntries;
            for (auto const& kw : intent["keywords"]) {
                GEODE_UNWRAP_INTO(auto phrases, kw["phrase"].as<std::vector<std::string>>());
                GEODE_UNWRAP_INTO(auto weight, kw["weight"].as<float>());
                keywordEntries.emplace_back(std::move(phrases), weight);
            }

            auto intentID = m_intentRegistry.intern(name);
            m_intentEngine.addIntent(intentID, std::move(keywordEntries));
            m_emotionModel.setIntentVAD(intentID, vad);

            if (name == "MATH_QUERY") m_mathQueryIntent = intentID;
            else if (name == "ENABLE_SETTING") m_enableSettingIntent = intentID;
            else if (name == "DISABLE_SETTING") m_disableSettingIntent = intentID;
            else if (name == "CHECK_SETTING_STATE") m_checkSettingStateIntent = intentID;
            else if (name == "COMPLAINT") m_complaintIntent = intentID;
        }

        // == Slot Schemas == //
        m_slotFiller.reset();
        m_slotRegistry.clear();
        for (auto const& schemaJson : json["slot_schemas"]) {
            GEODE_UNWRAP_INTO(auto intentName, schemaJson["intent"].as<std::string>());
            GEODE_UNWRAP_INTO(auto slots, schemaJson["slots"].as<std::vector<matjson::Value>>());

            SlotSchema schema;
            schema.intent = m_intentRegistry.intern(intentName);

            for (auto const& slotJson : slots) {
                GEODE_UNWRAP_INTO(auto slotName, slotJson["name"].as<std::string>());
                GEODE_UNWRAP_INTO(auto entityTypeName, slotJson["entity_type"].as<std::string>());
                GEODE_UNWRAP_INTO(auto required, slotJson["required"].as<std::optional<bool>>());
                GEODE_UNWRAP_INTO(auto expectedType, entityTypeFromString(entityTypeName));

                std::vector<std::string> prompts;
                auto promptValue = slotJson["prompt"];
                if (promptValue.isString()) {
                    GEODE_UNWRAP_INTO(auto prompt, promptValue.as<std::string>());
                    prompts.push_back(std::move(prompt));
                } else if (promptValue.isArray()) {
                    GEODE_UNWRAP_INTO(prompts, promptValue.as<std::vector<std::string>>());
                } else {
                    return geode::Err(fmt::format(
                        "Slot schema for intent '{}' slot '{}' has invalid prompt type",
                        intentName,
                        slotName
                    ));
                }

                if (prompts.empty()) {
                    prompts.push_back(fmt::format("Could you provide '{}' ?", slotName));
                }

                schema.slots.push_back({
                    .name = std::move(slotName),
                    .prompts = std::move(prompts),
                    .expectedType = expectedType,
                    .required = required.value_or(true),
                });
            }

            m_slotRegistry.registerSchema(std::move(schema));
        }

        // == Repetition == //
        if (!json["repetition"].isNull()) {
            GEODE_UNWRAP_INTO(auto decay, json["repetition"]["decay"].as<float>());
            GEODE_UNWRAP_INTO(auto historyLength, json["repetition"]["history_length"].as<size_t>());
            m_fatigueTracker = FatigueTracker(decay, historyLength);
        }

        // == Combos == //
        m_comboDetector.clear();
        for (auto const& combo : json["combos"]) {
            GEODE_UNWRAP_INTO(auto intents, combo["intents"].as<std::vector<std::string>>());
            GEODE_UNWRAP_INTO(auto vadOverride, combo["vad_override"].as<std::optional<std::string>>());
            auto tag = fmt::format("{}", fmt::join(intents, "+"));

            std::vector<Intent> intentIDs;
            for (auto const& intentName : intents) {
                intentIDs.push_back(m_intentRegistry.intern(intentName));
            }

            m_comboDetector.registerCombo({
                .key = { .intents = std::move(intentIDs) },
                .templateTag = std::move(tag),
                .vadOverride = vadOverride ? std::make_optional(m_intentRegistry.intern(*vadOverride)) : std::nullopt
            });
        }

        // == Templates == //
        m_templateEngine.clear();
        for (auto const& [key, slot] : json["slots"]) {
            SlotPool pool;

            for (auto const& entry : slot) {
                GEODE_UNWRAP_INTO(auto emotion, entry["emotion"].as<std::string>());
                GEODE_UNWRAP_INTO(auto fatigue, entry["fatigue"].as<std::string>());
                GEODE_UNWRAP_INTO(auto words, entry["words"].as<std::vector<std::string>>());

                auto emotionID = emotion == "*" ? Emotion(Emotion::INVALID) : m_emotionRegistry.intern(emotion);
                if (fatigue == "*") {
                    pool.add(emotionID, std::move(words));
                } else {
                    auto fatigueLevel = fatigueFromString(fatigue);
                    pool.add(emotionID, fatigueLevel, std::move(words));
                }
            }

            m_templateEngine.addSlot(key, std::move(pool));
        }

        for (auto const& [key, tmplt] : json["templates"]) {
            auto intent = m_intentRegistry.intern(key);
            for (auto const& entry : tmplt) {
                GEODE_UNWRAP_INTO(auto emotion, entry["emotion"].as<std::string>());
                GEODE_UNWRAP_INTO(auto fatigue, entry["fatigue"].as<std::string>());
                GEODE_UNWRAP_INTO(auto templates, entry["templates"].as<std::vector<std::string>>());

                auto emotionID = emotion == "*" ? Emotion(Emotion::INVALID) : m_emotionRegistry.intern(emotion);
                if (fatigue == "*") {
                    m_templateEngine.addTemplate(intent, emotionID, std::move(templates));
                } else {
                    auto fatigueLevel = fatigueFromString(fatigue);
                    m_templateEngine.addTemplate(intent, emotionID, fatigueLevel, std::move(templates));
                }
            }
        }

        for (auto const& [key, tmplt] : json["combo_templates"]) {
            for (auto const& entry : tmplt) {
                GEODE_UNWRAP_INTO(auto fatigue, entry["fatigue"].as<std::string>());
                GEODE_UNWRAP_INTO(auto templates, entry["templates"].as<std::vector<std::string>>());

                auto fatigueLevel = fatigueFromString(fatigue);
                m_templateEngine.addComboTemplate(key, fatigueLevel, std::move(templates));
            }
        }

        // == Settings == //
        auto addSettingAlias = [this](std::string alias, std::string id) {
            m_entityExtractor.addSetting(alias);
            m_settingNameMap.insert_or_assign(std::move(alias), std::move(id));
        };

        for (auto const& [name, value] : json["known_settings"]) {
            if (value.isString()) {
                GEODE_UNWRAP_INTO(auto id, value.as<std::string>());
                addSettingAlias(name, std::move(id));
                continue;
            }

            if (value.isArray()) {
                GEODE_UNWRAP_INTO(auto aliases, value.as<std::vector<std::string>>());
                for (auto const& alias : aliases) {
                    addSettingAlias(alias, name);
                }
                continue;
            }

            return geode::Err(fmt::format("Invalid setting entry for '{}'", name));
        }

        GEODE_UNWRAP_INTO(m_settingActualNames, json["setting_names"].as<geode::utils::StringMap<std::string>>());

        this->setupActions();

        return geode::Ok();
    }

    static bool looksLikeStateCheck(std::string_view input) {
        std::string lower(input);
        std::ranges::transform(lower, lower.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        bool queryLike = lower.find('?') != std::string::npos ||
            lower.rfind("is ", 0) == 0 ||
            lower.rfind("are ", 0) == 0 ||
            lower.find(" status") != std::string::npos ||
            lower.find(" state") != std::string::npos ||
            lower.find("currently") != std::string::npos;

        bool stateTerm = lower.find("enabled") != std::string::npos ||
            lower.find("disabled") != std::string::npos ||
            lower.find(" on") != std::string::npos ||
            lower.find(" off") != std::string::npos ||
            lower.find("status") != std::string::npos ||
            lower.find("state") != std::string::npos;

        return queryLike && stateTerm;
    }

    std::string Chatbot::process(std::string const& input) {
        auto entities = m_entityExtractor.extract(input);

        auto results = m_intentEngine.classify(input);
        Intent intent = Intent::INVALID;
        float conf = 0.f;
        if (!results.empty()) {
            intent = results[0].intent;
            conf = results[0].confidence;
        }

        for (auto const& e : entities) {
            if (e.type == EntityType::MATH_EXPRESSION) {
                intent = m_mathQueryIntent;
                conf = std::max(conf, 0.85f);
                break;
            }
        }

        if (
            m_checkSettingStateIntent &&
            looksLikeStateCheck(input) &&
            std::ranges::any_of(entities, [](Entity const& e) {
                return e.type == EntityType::SETTING_NAME;
            })
        ) {
            intent = m_checkSettingStateIntent;
            conf = std::max(conf, 0.9f);
        }

        intent = this->resolveIntent(intent, entities);

        auto hasSettingEntity = [&entities]() {
            return std::ranges::any_of(entities, [](Entity const& e) {
                return e.type == EntityType::SETTING_NAME;
            });
        };

        auto isSettingIntent = [this](Intent i) {
            return i == m_enableSettingIntent ||
                i == m_disableSettingIntent ||
                i == m_checkSettingStateIntent;
        };

        if (isSettingIntent(intent) && !hasSettingEntity()) {
            if (auto recent = m_context.findRecentEntity(EntityType::SETTING_NAME, 4)) {
                entities.push_back(*recent);
            }
        }

        std::optional<std::string> actionOutput;
        bool actionFailed = false;
        SlotFillResult fillResult;
        if (m_slotFiller.isActive() && intent && intent != m_slotFiller.activeIntent()) {
            fillResult = m_slotFiller.begin(intent, entities);
        } else if (m_slotFiller.isActive()) {
            fillResult = m_slotFiller.continueFill(entities);
        } else {
            fillResult = m_slotFiller.begin(intent, entities);
        }

        if (fillResult.status == SlotFillStatus::COMPLETE) {
            m_slotFiller.reset();
            if (auto actionResult = m_actionRegistry.dispatch(fillResult.intent, fillResult.filled)) {
                actionOutput = actionResult->response;
                actionFailed = !actionResult->success;
            }
        }

        m_context.addEntry({ input, entities, intent });

        auto fatigueLevel = m_fatigueTracker.record(intent, input);

        m_emotionModel.applyIntent(intent, conf);
        auto vadDelta = FatigueTracker::fatigueVadDelta(fatigueLevel);
        m_emotionModel.nudge(vadDelta);
        if (actionFailed) {
            m_emotionModel.applyIntent(m_complaintIntent, 0.5f);
        }
        Emotion emotion = m_emotionModel.currentEmotion();

        std::string reply;
        try {
            reply = callLMStudio(
                input,
                emotion,
                static_cast<float>(fatigueLevel),
                recentContextMessages(m_context, 3),
                actionOutput
            );
        } catch (...) {
            geode::log::warn("LM Studio request failed with an unexpected exception");
            reply = LM_STUDIO_FALLBACK;
        }

        m_emotionModel.decay(m_emotionDecay);
        m_fatigueTracker.decay();

        if (isBlank(reply)) {
            return LM_STUDIO_FALLBACK;
        }

        return reply;
    }

    void Chatbot::setupActions() {
        if (m_mathQueryIntent) {
            m_actionRegistry.registerAction(m_mathQueryIntent, [](EntityMap const& slots) -> ActionResult {
                auto it = slots.find("expression");
                if (it == slots.end()) {
                    return { "I still need the expression to evaluate.", false };
                }

                auto res = rift::evaluate(it->second.raw);
                if (!res) {
                    return { "Sorry, I'm bad at math", false };
                }

                return { fmt::format("The answer is {}.", *res), true };
            });
        }

        if (m_enableSettingIntent) {
            m_actionRegistry.registerAction(
                m_enableSettingIntent,
                [this](EntityMap const& slots) -> ActionResult {
                    auto setting = resolveSettingID(slots, m_settingNameMap);
                    if (!setting) {
                        return { "What setting would you like to enable?", false };
                    }

                    auto actualIt = m_settingActualNames.find(*setting);
                    std::string actualName = actualIt != m_settingActualNames.end() ? actualIt->second : *setting;

                    if (config::get<bool>(*setting, false)) {
                        return { fmt::format("{} is already enabled.", actualName), true };
                    }

                    config::set(*setting, true);

                    return { fmt::format("{} enabled.", actualName), true };
                }
            );
        }

        if (m_disableSettingIntent) {
            m_actionRegistry.registerAction(
                m_disableSettingIntent,
                [this](EntityMap const& slots) -> ActionResult {
                    auto setting = resolveSettingID(slots, m_settingNameMap);
                    if (!setting) {
                        return { "What setting would you like to disable?", false };
                    }

                    auto actualIt = m_settingActualNames.find(*setting);
                    std::string actualName = actualIt != m_settingActualNames.end() ? actualIt->second : *setting;

                    if (!config::get<bool>(*setting, false)) {
                        return { fmt::format("{} is already disabled.", actualName), true };
                    }

                    config::set(*setting, false);

                    return { fmt::format("{} disabled.", actualName), true };
                }
            );
        }

        if (m_checkSettingStateIntent) {
            m_actionRegistry.registerAction(
                m_checkSettingStateIntent,
                [this](EntityMap const& slots) -> ActionResult {
                    auto setting = resolveSettingID(slots, m_settingNameMap);
                    if (!setting) {
                        return { "Which setting's state would you like to check?", false };
                    }

                    auto actualIt = m_settingActualNames.find(*setting);
                    std::string actualName = actualIt != m_settingActualNames.end() ? actualIt->second : *setting;

                    bool enabled = config::get<bool>(*setting, false);
                    return { fmt::format("{} is currently {}.", actualName, enabled ? "enabled" : "disabled"), true };
                }
            );
        }
    }

    Intent Chatbot::resolveIntent(Intent intent, std::span<Entity const> entities) const {
        if (
            m_mathQueryIntent == Intent::INVALID &&
            m_enableSettingIntent == Intent::INVALID &&
            m_disableSettingIntent == Intent::INVALID &&
            m_checkSettingStateIntent == Intent::INVALID
        ) {
            return intent;
        }

        if (intent) {
            return intent;
        }

        auto hasEntityType = [entities](EntityType type) {
            return std::ranges::any_of(entities, [type](Entity const& e) {
                return e.type == type;
            });
        };

        bool hasMathExpression = hasEntityType(EntityType::MATH_EXPRESSION);
        bool hasSettingName = hasEntityType(EntityType::SETTING_NAME);

        auto resolveSettingFromRecentIntent = [this, hasSettingName](Intent intent) -> Intent {
            if (!hasSettingName || !intent || !m_context.containsIntent(intent, 2)) {
                return Intent::INVALID;
            }
            if (m_checkSettingStateIntent) {
                return m_checkSettingStateIntent;
            }
            return intent;
        };

        if (m_mathQueryIntent && hasMathExpression && m_context.containsIntent(m_mathQueryIntent, 2)) {
            return m_mathQueryIntent;
        }

        if (Intent resolvedEnable = resolveSettingFromRecentIntent(m_enableSettingIntent)) {
            return resolvedEnable;
        }

        if (Intent resolvedDisable = resolveSettingFromRecentIntent(m_disableSettingIntent)) {
            return resolvedDisable;
        }

        if (m_checkSettingStateIntent && hasSettingName) {
            return m_checkSettingStateIntent;
        }

        return intent;
    }
}
