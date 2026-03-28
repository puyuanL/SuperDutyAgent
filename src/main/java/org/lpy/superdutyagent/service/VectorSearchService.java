package org.lpy.superdutyagent.service;

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
import com.alibaba.cloud.ai.model.RerankModel;
import com.alibaba.cloud.ai.model.RerankRequest;
import com.alibaba.cloud.ai.model.RerankResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

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

    @Autowired
    private RerankModel rerankModel;

    @Value("${spring.ai.dashscope.rerank.top-n}")
    private int rerankTopN;

    /**
     * 搜索相似文档
     * 召回 + 重拍
     * @param query 查询文本
     * @param topK 返回最相似的K个结果
     * @return 搜索结果列表
     */
     public List<SearchResult> searchSimilarDocuments(String query, int topK) {
         // 1. 召回数据库向量
         List<SearchResult> results = recallDatabase(query, topK);

         // 2. 使用重排模型对结果进行重排序
         if (!results.isEmpty()) {
             results = rerankResults(query, results);
             logger.info("重排完成, 返回 {} 个文档", results.size());
         }

         return results;
    }

    /**
     * query 召回向量
     *
     * @param query 查询文本
     * @param topK 返回最相似的K个结果
     * @return 搜索结果列表
     */
    private List<SearchResult> recallDatabase(String query, int topK) {
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

            logger.info("向量搜索完成, 找到 {} 个相似文档", results.size());
            return results;

        } catch (Exception e) {
            logger.error("搜索相似文档失败", e);
            throw new RuntimeException("搜索失败: " + e.getMessage(), e);
        }
    }

    /**
     * 使用重排模型对搜索结果进行重排序
     *
     * @param query   原始查询
     * @param results 向量搜索结果
     * @return 重排后的结果
     */
    private List<SearchResult> rerankResults(String query, List<SearchResult> results) {
        try {
            // 将 SearchResult 转换为 Spring AI Document
            List<Document> documents = results.stream()
                    .map(result -> new Document(
                            result.getId(),
                            result.getContent(),
                            result.getMetadata() != null ?
                                    Map.of("metadata", result.getMetadata()) : Map.of()
                    ))
                    .collect(Collectors.toList());

            // 创建重排请求
            RerankRequest rerankRequest = new RerankRequest(query, documents);

            // 调用重排模型
            RerankResponse rerankResponse = rerankModel.call(rerankRequest);

            // 获取重排结果并过滤低于阈值的文档
            List<SearchResult> rerankedResults = rerankResponse.getResults().stream()
                    .map(doc -> {
                        SearchResult result = new SearchResult();
                        result.setId(doc.getOutput().getId());
                        result.setContent(doc.getOutput().getText());
                        result.setScore(Double.valueOf(doc.getScore()).floatValue());
                        Object metadata = doc.getMetadata();
                        if (metadata != null) {
                            result.setMetadata(metadata.toString());
                        }
                        return result;
                    })
                    .collect(Collectors.toList());

            logger.debug("重排模型返回 {} 个结果", rerankedResults.size());
            return rerankedResults;

        } catch (Exception e) {
            logger.error("重排失败，返回原始结果 top-n", e);
            // 重排失败时返回原始结果
            return results.stream().limit(rerankTopN).collect(Collectors.toList());
        }
    }

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
