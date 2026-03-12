package org.lpy.superdutyagent.service;

import com.alibaba.cloud.ai.model.RerankModel;
import com.alibaba.cloud.ai.model.RerankRequest;
import com.alibaba.cloud.ai.model.RerankResponse;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.SearchResults;
import io.milvus.param.R;
import io.milvus.param.dml.SearchParam;
import io.milvus.response.SearchResultsWrapper;
import lombok.Getter;
import lombok.Setter;
import org.lpy.superdutyagent.constant.MilvusConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * 向量搜索服务
 * 负责从 Milvus 中搜索相似向量
 */
@Service
public class VectorSearchService {

    private static final Logger logger = LoggerFactory.getLogger(VectorSearchService.class);

    @Autowired
    private MilvusServiceClient milvusClient;

    @Autowired
    private VectorEmbeddingService embeddingService;

//    @Autowired
//    private RerankModel rerankModel;

    /**
     * 搜索相似文档
     * 
     * @param query 查询文本
     * @param topK 返回最相似的K个结果
     * @return 搜索结果列表
     */
    public List<SearchResult> searchSimilarDocuments(String query, int topK) {
        try {
            logger.info("开始搜索相似文档, 查询: {}, topK: {}", query, topK);

            // 1. 将查询文本向量化
            List<Float> queryVector = embeddingService.generateQueryVector(query);
            logger.debug("查询向量生成成功, 维度: {}", queryVector.size());

            // 2. 构建搜索参数
            SearchParam searchParam = SearchParam.newBuilder()
                    .withCollectionName(MilvusConstants.MILVUS_COLLECTION_NAME)
                    .withVectorFieldName("vector")
                    .withVectors(Collections.singletonList(queryVector))
                    .withTopK(topK)
                    .withMetricType(io.milvus.param.MetricType.L2)
                    .withOutFields(List.of("id", "content", "metadata"))
                    .withParams("{\"nprobe\":10}")
                    .build();

            // 3. 执行搜索
            R<SearchResults> searchResponse = milvusClient.search(searchParam);

            if (searchResponse.getStatus() != 0) {
                throw new RuntimeException("向量搜索失败: " + searchResponse.getMessage());
            }

            // 4. 解析搜索结果
            SearchResultsWrapper wrapper = new SearchResultsWrapper(searchResponse.getData().getResults());
            List<SearchResult> results = new ArrayList<>();

            for (int i = 0; i < wrapper.getRowRecords(0).size(); i++) {
                SearchResult result = new SearchResult();
                result.setId((String) wrapper.getIDScore(0).get(i).get("id"));
                result.setContent((String) wrapper.getFieldData("content", 0).get(i));
                result.setScore(wrapper.getIDScore(0).get(i).getScore());
                
                // 解析 metadata
                Object metadataObj = wrapper.getFieldData("metadata", 0).get(i);
                if (metadataObj != null) {
                    result.setMetadata(metadataObj.toString());
                }
                
                results.add(result);
            }

            logger.info("搜索完成, 找到 {} 个相似文档", results.size());
            return results;

        } catch (Exception e) {
            logger.error("搜索相似文档失败", e);
            throw new RuntimeException("搜索失败: " + e.getMessage(), e);
        }
    }

//    /**
//     * 召回：仅通过Milvus向量检索获取topK条相似文档（粗排）
//     * @param query 查询文本
//     * @param topK  召回的数量（建议大于最终需要的数量，如最终要10条则召回20-30条）
//     * @return 向量检索结果列表（按向量相似度排序）
//     */
//    public List<SearchResult> recallDocuments(String query, int topK) {
//        try {
//            logger.info("开始召回相似文档, 查询: {}, 召回数量: {}", query, topK);
//
//            // 1. 生成查询向量
//            List<Float> queryVector = embeddingService.generateQueryVector(query);
//            logger.debug("查询向量生成成功, 维度: {}", queryVector.size());
//
//            // 2. 构建Milvus搜索参数
//            SearchParam searchParam = SearchParam.newBuilder()
//                    .withCollectionName(MilvusConstants.MILVUS_COLLECTION_NAME)
//                    .withVectorFieldName("vector")
//                    .withVectors(Collections.singletonList(queryVector))
//                    .withTopK(topK)
//                    .withMetricType(io.milvus.param.MetricType.L2)
//                    .withOutFields(List.of("id", "content", "metadata"))
//                    .withParams("{\"nprobe\":10}")
//                    .build();
//
//            // 3. 执行向量召回
//            R<SearchResults> searchResponse = milvusClient.search(searchParam);
//            if (searchResponse.getStatus() != 0) {
//                throw new RuntimeException("Milvus向量召回失败: " + searchResponse.getMessage());
//            }
//
//            // 4. 解析召回结果
//            SearchResultsWrapper wrapper = new SearchResultsWrapper(searchResponse.getData().getResults());
//            List<SearchResult> recallResults = new ArrayList<>();
//
//            for (int i = 0; i < wrapper.getRowRecords(0).size(); i++) {
//                SearchResult result = new SearchResult();
//                result.setId((String) wrapper.getIDScore(0).get(i).get("id"));
//                result.setContent((String) wrapper.getFieldData("content", 0).get(i));
//                // 向量召回得分（L2距离，值越小越相似）
//                result.setScore(wrapper.getIDScore(0).get(i).getScore());
//
//                // 解析metadata
//                Object metadataObj = wrapper.getFieldData("metadata", 0).get(i);
//                if (metadataObj != null) {
//                    result.setMetadata(metadataObj.toString());
//                }
//
//                recallResults.add(result);
//            }
//
//            logger.info("召回完成, 共召回 {} 条相似文档", recallResults.size());
//            return recallResults;
//
//        } catch (Exception e) {
//            logger.error("召回相似文档失败", e);
//            throw new RuntimeException("召回失败: " + e.getMessage(), e);
//        }
//    }
//
//    /**
//     * 重排：使用 Cross Encoder (DashScope Rerank) 对召回结果进行精排
//     * @param query 查询文本
//     * @param recallResults 向量召回的原始结果
//     * @param finalTopK 最终返回的结果数量
//     * @return 重排后的结果列表（按 Cross Encoder 分数降序）
//     */
//    public List<SearchResult> rerankDocuments(String query, List<SearchResult> recallResults, int finalTopK) {
//        try {
//            logger.info("开始重排, Query: {}, 召回数量: {}, 目标TopK: {}", query, recallResults.size(), finalTopK);
//
//            if (recallResults == null || recallResults.isEmpty()) {
//                logger.warn("召回结果为空，直接返回空列表");
//                return new ArrayList<>();
//            }
//
//            // 1. 转换召回结果为 Spring AI 标准 Document 列表（保留原始索引）
//            List<Document> documents = new ArrayList<>();
//            for (int i = 0; i < recallResults.size(); i++) {
//                SearchResult result = recallResults.get(i);
//                // 构建 Document，将原始索引存入 metadata，方便后续关联
//                Document doc = new Document(result.getContent());
//                doc.getMetadata().put("originalIndex", i); // 关键：记录原始召回结果的下标
//                doc.getMetadata().put("id", result.getId()); // 携带原始ID
//                doc.getMetadata().put("metadata", result.getMetadata()); // 携带原始元数据
//                documents.add(doc);
//            }
//
//            // 2. 构建重排请求（兼容无 builder 的版本）
//            // 方式1：直接构造 RerankRequest（替代 builder 模式）
//            RerankRequest request = new RerankRequest(query, documents, recallResults.size());
//
//            // 3. 调用 Cross Encoder 重排模型
//            RerankResponse response = rerankModel.call(request);
//
//            // 4. 解析重排结果，映射回 SearchResult 并排序
//            List<SearchResult> rankedResults = new ArrayList<>();
//            for (RerankedDocument rerankedDoc : response.getRerankedDocuments()) {
//                // 获取原始召回结果的下标（从 Document 的 metadata 中取出）
//                int originalIndex = (int) rerankedDoc.getDocument().getMetadata().get("originalIndex");
//                SearchResult originalResult = recallResults.get(originalIndex);
//
//                // 构建重排后的 SearchResult（替换为 Cross Encoder 分数）
//                SearchResult rerankedResult = new SearchResult();
//                rerankedResult.setId(originalResult.getId());
//                rerankedResult.setContent(originalResult.getContent());
//                rerankedResult.setMetadata(originalResult.getMetadata());
//                // Cross Encoder 分数（0~1 之间，越高越相关）
//                rerankedResult.setScore((float) rerankedDoc.getScore());
//
//                rankedResults.add(rerankedResult);
//            }
//
//            // 5. 截取最终需要的 TopK（重排结果本身已按分数降序排列）
//            if (rankedResults.size() > finalTopK) {
//                rankedResults = rankedResults.subList(0, finalTopK);
//            }
//
//            logger.info("重排完成，返回 {} 条结果", rankedResults.size());
//            return rankedResults;
//
//        } catch (Exception e) {
//            logger.error("重排过程出错", e);
//            throw new RuntimeException("重排失败: " + e.getMessage(), e);
//        }
//    }

    /**
     * 搜索结果类
     */
    @Setter
    @Getter
    public static class SearchResult {
        private String id;
        private String content;
        private float score;
        private String metadata;

    }
}
