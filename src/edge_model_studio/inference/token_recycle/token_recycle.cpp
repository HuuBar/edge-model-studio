/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Description: http server
 * Create: 2025-04-25
 */

#include <cstddef>
#include <ctime>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cctype>
#include <algorithm>
#include "token_recycle.h"

#define MS_TIME 1000000
#define WEIGHT_OFFSET 10000
#define WEIGHT_SIZE 1024
#define TIME_RATIO 1000.0
#define PROMPT_FOR_SHOW_LEN 100

namespace {
std::mutex dlliteMtx;
const size_t MAX_LEFTOVER = 4;
const size_t RETRY_DELAY_TIME = 2;
const uint32_t treeIdx = 0;
////58节点深度4
//constexpr uint32_t gamma = 4;
//const uint32_t topKSpecDim = 16;
//uint32_t topHead[gamma] = {1,6,13,6};
//uint32_t topKSpec[gamma][16] = {{6}, {6, 5, 3, 2, 1, 1}, {5, 3, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1}, {3, 2, 1, 1, 1, 1}};

////32节点深度4
//constexpr uint32_t gamma = 4;
//const uint32_t topKSpecDim = 16;
//uint32_t topHead[gamma] = {1,6,7,5};
//uint32_t topKSpec[gamma][16] = {{6}, {3, 2, 2, 1, 1, 1}, {2, 2, 2, 2, 1, 1, 1}, {1, 1, 1, 1, 1}};

////24节点深度4
//constexpr uint32_t gamma = 4;
//const uint32_t topKSpecDim = 16;
//uint32_t topHead[gamma] = {1, 6, 4, 2};
//uint32_t topKSpec[gamma][16] = {{6}, {3, 2, 2, 1, 1, 1}, {2, 2, 1, 1}, {1, 1}};

////16节点深度4
//constexpr uint32_t gamma = 4;
//const uint32_t topKSpecDim = 16;
//uint32_t topHead[gamma] = {1, 4, 3, 1};
//uint32_t topKSpec[gamma][16] = {{6}, {2, 2, 1, 1}, {1, 1, 1}, {1}};

//12节点深度3
constexpr uint32_t gamma = 3;
const uint32_t topKSpecDim = 16;
uint32_t topHead[3] = {1,2,2};
uint32_t topKSpec[gamma][16] = {{6}, {2, 2}, {1, 1}};

////6节点深度4
//constexpr uint32_t gamma = 4;
//const uint32_t topKSpecDim = 16;
//uint32_t topHead[4] = {1,2,1,1};
//uint32_t topKSpec[gamma][16] = {{2}, {1, 1}, {1}, {1}};

////8节点深度4
//constexpr uint32_t gamma = 4;
//const uint32_t topKSpecDim = 16;
//uint32_t topHead[4] = {1,2,1,1};
//uint32_t topKSpec[gamma][16] = {{4}, {1, 1}, {1}, {1}};

std::string ExtractPrompt(const std::string& llm_engine_prompt) {
    const std::string start_token = "<|im_start|>user";
    const std::string end_token = "<|im_end|>";

    size_t start_pos = llm_engine_prompt.rfind(start_token);
    if (start_pos == std::string::npos) {
        return "";
    }

    size_t end_pos = llm_engine_prompt.find(end_token, start_pos);
    if (end_pos == std::string::npos) {
        return "";
    }

    std::string prompt = llm_engine_prompt.substr(
        start_pos + start_token.size(),
        end_pos - start_pos - start_token.size()
    );

    auto trim = [](std::string& s) {
        size_t first = s.find_first_not_of(" \n\r\t");
        size_t last = s.find_last_not_of(" \n\r\t");
        if (first == std::string::npos || last == std::string::npos) {
            s.clear();
        } else {
            s = s.substr(first, last - first + 1);
        }
    };
    trim(prompt);

    return prompt;
}

double GetTimeInUs()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<double>(tv.tv_sec * MS_TIME + tv.tv_usec);
}
}

std::unordered_map<const LLMEngine_Context *, Session *> g_sessionsMap;
std::unordered_map<const LLMEngine_Context *, std::string> g_promptMap;
static std::mutex g_promptMtx;

void InitDefaultTopK()
{
    token_topk_table_ = std::make_unique<std::vector<TopKPair>>(TOKENIZER_SIZE);

    for (auto& row : *token_topk_table_) {
        for (auto& p : row) {
            p.first = UINT32_MAX;
            p.second = -std::numeric_limits<float>::infinity();
        }
    }

    LOGI("[Session] Allocated default token_topk_table_") ;
}

bool LoadTopKTable(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        LOGE("[Session] LoadTopKTable failed: cannot open file") ;
        return false;
    }

    token_topk_table_ = std::make_unique<std::vector<TopKPair>>(TOKENIZER_SIZE);

    ifs.read(reinterpret_cast<char*>(token_topk_table_->data()),
                token_topk_table_->size() * sizeof(TopKPair));

    if (!ifs) {
        LOGE("[Session] LoadTopKTable failed: file read error");
        token_topk_table_.reset();
        return false;
    }
    LOGI("[Session] Loaded token_topk_table_ from %{public}s", path.c_str()) ;
    return true;
}

bool SaveTopKTable(const std::string& path) const
{
    if (!token_topk_table_) {
        LOGE("[Session] SaveTopKTable: table not initialized");
        return false;
    }

    try {
        std::filesystem::path p(path);
        auto parent = p.parent_path();

        if (!parent.empty() && !std::filesystem::exists(parent)) {
            LOGI("[Session] SaveTopKTable: creating directory %s",
                      parent.string().c_str());

            if (!std::filesystem::create_directories(parent)) {
                LOGE("[Session] SaveTopKTable: failed to create directory %s",
                          parent.string().c_str());
                return false;
            }
        }
    } catch (const std::exception& e) {
        LOGE("[Session] SaveTopKTable: exception creating directory: %s", e.what());
        return false;
    }

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        LOGE("[Session] SaveTopKTable: cannot open file %s", path.c_str());
        return false;
    }

    ofs.write(reinterpret_cast<const char*>(token_topk_table_->data()),
              token_topk_table_->size() * sizeof(TopKPair));

    if (!ofs.good()) {
        LOGE("[Session] SaveTopKTable: write failed");
        return false;
    }

    LOGI("[Session] Saved token_topk_table_ to %s", path.c_str());
    return true;
}

void InitTopKTable(const std::string& path)
{
    if (token_topk_table_) return;

    if (LoadTopKTable(path)) {
        return;
    }
    InitDefaultTopK();
}

void UpdateTokenTopkTable(const uint32_t draftTokenLen, const std::vector<uint32_t>& tokens, const std::vector<uint32_t>& indices, const std::vector<float>& logits)
{
   for (size_t i = 0; i < draftTokenLen; ++i) {
       uint32_t token = tokens[i];
       auto& table_row = (*token_topk_table_)[token];
       size_t base = i * TOPK;
       for (int k = 0; k < TOPK; ++k) {
           table_row[k].first  = indices[base + k];
           table_row[k].second = logits[base + k];
       }
   }
}

void OnPrefillGenerateDone(const LLMEngine_Context* ctx, const std::string& prompt)
{
    (void)ctx;
    uint32_t len = prompt.size() * TOPK;
    std::vector<uint32_t> indices(len);
    std::vector<float> logits(len);
    std::vector<int> prompt_ids;
    std::string tokenizer_path = "";
    InitTopKTable("model_path");
    GLMInitParams_.tokenizerPath = tokenizer_path;
    auto ret = tokenizer_.Init(GLMInitParams_);
    if (ret != SUCCESS) {
        LOGE("Tokenizer init failed.");
        return;
    }
    auto status = tokenizer_.Encode(prompt, prompt_ids);

    if (status != SUCCESS) {
        LOGE("Encode prompt failed");
        return;
    }
    
    if (prompt_ids.empty()) {
        LOGI("prompt_ids is empty");
        return;
    }
    
    std::string decoded;
    status = tokenizer_.Decode(prompt_ids, decoded);
    
    LOGI("zpfprompt_idtotoken is %{public}s", decoded.c_str());

    LOGI("prompt_ids are [");
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        if (i != 0) LOGI(", ");
        LOGI("prompt_id%{public}d", prompt_ids[i]);
        if (prompt_ids[i] > TOKENIZER_SIZE || prompt_ids[i] < 0) {
            LOGI("zpf prompt_id out of range");
        }
    }
    LOGI("]");
    LOGI("prefill len is :%{public}d", len);
    ret = LLMEngine_Context_GetPrefillAllTokenGeneration(ctx, len, indices.data(), logits.data());
    if (ret != SUCCESS) {
        LOGE("Get prefill generation info failed.");
        return;
    }
    LOGE("Get prefill generation info successed.");
    uint32_t effective_len = prompt_ids.size();
    for (uint32_t i = 0; i < effective_len; ++i) {
        int token_id = prompt_ids[i];
        if (token_id < 0 || token_id >= (int)TOKENIZER_SIZE)
            continue;

        for (uint32_t k = 0; k < TOPK; ++k) {
            uint32_t flat_idx = i * TOPK + k;
            (*token_topk_table_)[token_id][k] = std::make_pair(indices[flat_idx], logits[flat_idx]);
        }
    }

    LOGI("Updated %{public}u tokens' topk entries", len);
}

void onPrefillGenerateDoneFunc(const LLMEngine_Context* ctx)
{
    Session *session = FindSessionByCtx(ctx);
    if (session == nullptr) {
        LOGE("failed to find session by context.");
        return;
    }
    std::string prompt;
    {
        std::lock_guard<std::mutex> lk(g_promptMtx);
        auto it = g_promptMap.find(ctx);
        if (it != g_promptMap.end()) {
            prompt = it->second;
            g_promptMap.erase(it);
        } else {
            LOGW("Prompt not found for ctx %p", ctx);
        }
    }
    session->OnPrefillGenerateDone(ctx, prompt);
};

void onGenerateDraftTokensFunc(const LLMEngine_Context* ctx, uint32_t* draftTokenIds, uint32_t* draftTokenLen, int32_t* acceptIds, uint32_t acceptLen)
{
    Session *session = FindSessionByCtx(ctx);
    if (session == nullptr) {
        LOGE("failed to find session by context.");
        return;
    }
    session->OnGenerateDraftTokens(ctx, draftTokenIds, draftTokenLen, acceptIds, acceptLen);
};

void OnGenerateDraftTokens(const LLMEngine_Context* ctx, uint32_t* draftTokenIds, uint32_t* draftTokenLen, int32_t* acceptIds, uint32_t acceptLen)
{
    auto draftTokenStart = std::chrono::high_resolution_clock::now();
    std::vector<int> acceptIdsVec;
    acceptIdsVec.reserve(acceptLen);
    for (int i = 0; i < acceptLen; i++) {
        acceptIdsVec.push_back(static_cast<int>(acceptIds[i]));
        LOGI("acceptIds %{public}u : %{public}u", i, acceptIds[i]);
    }
    uint32_t len = 64 * TOPK;
    std::vector<uint32_t> last_draft_ids(64);
    std::vector<uint32_t> indices(len);
    std::vector<float> logits(len);
    uint32_t tokenIdLen = 0;
    if (acceptLen == 0) {
        LOGI("First decode or no lastGenerationResult, draft disabled.");
        *draftTokenLen = 0;
         return;
    }
    int32_t generatedLastTokenId = acceptIds[acceptLen -1];

    auto ret = LLMEngine_Context_GetDecodeGeneration(ctx, len, indices.data(), logits.data(), last_draft_ids.data(), &tokenIdLen);
    if (ret != 0) {
        LOGE("GetDecodeGeneration failed");
        return;
    }
    LOGI("GetDecodeGeneration Done, tokenIdLen=%{public}u", tokenIdLen);
    
    std::vector<int> last_generation_ids;
    std::string test_last_generation_ids;
    last_generation_ids.push_back(generatedLastTokenId);
    LOGI("zpf generatedLastTokenId is: %{public}d", generatedLastTokenId);
    auto status = tokenizer_.Decode(last_generation_ids, test_last_generation_ids);
    LOGI("test_last_generation_ids is: %{public}s", test_last_generation_ids.c_str());
    if (status != SUCCESS || test_last_generation_ids.empty()) {
        LOGE("Decode lastGenerationResult_ failed");
        return;
    }
    
    std::string ids_str;
    for (int i = 0; i < tokenIdLen; i++) {
        if (i > 0) ids_str += ", ";
        ids_str += std::to_string(last_draft_ids[i]);
    }
    LOGI("zpf last_draft_ids=[%{public}s]", ids_str.c_str()); //to do
    
    std::string last_draft_tokens;
    std::vector<int> zpf_draft_ids;
    for (int i = 0; i < last_draft_ids.size(); i++) {
        zpf_draft_ids.push_back(static_cast<int>(last_draft_ids[i]));
    }
    status = tokenizer_.Decode(zpf_draft_ids, last_draft_tokens);
    LOGI("zpf last_draft_tokens is: %{public}s", last_draft_tokens.c_str());
    
    if (tokenIdLen != 0) {
        UpdateTokenTopkTable(tokenIdLen, last_draft_ids, indices, logits);
        LOGI("zpfUpdateTokenTopkTable Done");
    }

    std::vector<uint32_t> flat_draft;

    uint32_t root_token = generatedLastTokenId;//to do
    flat_draft.push_back(root_token);
    std::vector<std::vector<uint32_t>> layers;
    layers.resize(gamma + 1);
    layers[0] = {root_token};

    for (uint32_t layer = 0; layer < gamma; ++layer) {
        const auto& current_parents = layers[layer];
        auto& next_layer = layers[layer + 1];
    
        uint32_t nodes_to_expand = std::min<uint32_t>(topHead[layer], static_cast<uint32_t>(current_parents.size()));
    
        for (uint32_t h = 0; h < nodes_to_expand; ++h) {
            uint32_t parent_token = current_parents[h]; // 只取前 topHead[layer] 个
    
            if (parent_token >= TOKENIZER_SIZE) {
                LOGE("FATAL: parent_token %u out of range!", parent_token);
                continue;
            }
    
            const auto& topk_list = (*token_topk_table_)[parent_token];
            uint32_t num_children = std::min<uint32_t>(topKSpec[layer][h], TOPK);
    
            for (uint32_t k = 0; k < num_children; ++k) {
                uint32_t child_token = (topk_list[k].first < TOKENIZER_SIZE) ? topk_list[k].first : 0u;
                flat_draft.push_back(child_token);
                next_layer.push_back(child_token);
            }
        }
    }

    LOGI("zpf flat_draft build finished! total nodes = %{public}zu (include root)", flat_draft.size());

    if (flat_draft.size() <= 1) {
        LOGW("zpf draft tree empty or only root, skip speculative decoding");
    } else {
        *draftTokenLen = flat_draft.size() - 1;
        memcpy(draftTokenIds, flat_draft.data() + 1, (*draftTokenLen) * sizeof(uint32_t));
        std::vector<int> token_ids;
        token_ids.reserve(flat_draft.size() - 1);
    
        for (int i = 1; i < flat_draft.size(); ++i) {
            token_ids.push_back(static_cast<int>(flat_draft[i]));
        }
        std::ostringstream oss;
        for (size_t i = 0; i < token_ids.size(); ++i) {
            if (i > 0) oss << ",";
            oss << token_ids[i];
        }
        std::string result = oss.str();
        LOGI("zpf draft_token : %{public}s ", result.c_str());
        
        std::ostringstream per_token_oss_with_trailing_comma;
        for (size_t i = 0; i < token_ids.size(); ++i) {
            std::vector<int> single_token = {token_ids[i]};
            std::string decoded;
            auto status = tokenizer_.Decode(single_token, decoded);
            per_token_oss_with_trailing_comma << decoded << ", ";
        }
        std::string per_token_str = per_token_oss_with_trailing_comma.str();
        
        LOGI("zpf draft_tokens per token: %{public}s", per_token_str.c_str());
    
        std::string draftTokenStr;
        status = tokenizer_.Decode(token_ids, draftTokenStr);
        if (status != SUCCESS) {
            LOGE("Decode draftTokenStr failed, status=%d", status);
        } else {
            LOGI("Successfully generated draft: %{public}s (tokens=%{public}zu)", 
                      draftTokenStr.c_str(), token_ids.size());
        }
    }
    auto draftTokenEnd = std::chrono::high_resolution_clock::now();
    auto draftDurationMs = std::chrono::duration_cast<std::chrono::microseconds>(draftTokenEnd - draftTokenStart).count();
    LOGI("zpf generate draft time cost [%{public}lld ns]", draftDurationMs);
}