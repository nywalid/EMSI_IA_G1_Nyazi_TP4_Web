package ma.emsi.nyazi.tp4_web_nyazi.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.enterprise.context.Dependent;

import java.io.InputStream;
import java.io.Serializable;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


@Dependent
public class LlmClientRoutage implements Serializable {
    private EmbeddingStore<TextSegment> creerEmbeddingStore(String Fichier) {
        InputStream is = getClass().getClassLoader().getResourceAsStream(Fichier);
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = documentParser.parse(is);
        DocumentSplitter splitter = DocumentSplitters.recursive(500, 50);

        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel= new AllMiniLmL6V2EmbeddingModel();
        Response<List<Embedding>> response= embeddingModel.embedAll(segments);
        List<Embedding> listeEmbedding= response.content();

        EmbeddingStore<TextSegment> embeddingStore= new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(listeEmbedding, segments);

        return embeddingStore;
    }
    public interface Assistant {
        String chat(String prompt);
    }

    private String systemRole;
    private ChatMemory chatMemory;
    private Assistant assistant;

    public LlmClientRoutage() {

        String cle= System.getenv("GEMINI_KEY");
        ChatModel modele = GoogleAiGeminiChatModel
                .builder()
                .apiKey(cle)
                .modelName("gemini-2.5-flash")
                .logRequestsAndResponses(true)
                .temperature(0.7)
                .build();

        EmbeddingStore<TextSegment> storeRAG = creerEmbeddingStore("rag.pdf");
        EmbeddingStore<TextSegment> storeEMSI = creerEmbeddingStore("Brochure_EMSI.pdf");


        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        ContentRetriever retrieverRAG = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeRAG)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        ContentRetriever retrieverEMSI = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeEMSI)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(retrieverRAG, "Documents techniques sur l'intelligence artificielle, RAG, embeddings, modèles généraux.");
        descriptions.put(retrieverEMSI, "Document de EMSI, formations et parcours");


        QueryRouter router = new LanguageModelQueryRouter(modele, descriptions);

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(modele)
                .retrievalAugmentor(augmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }

    /** Setter appelé une seule fois : au début du chat */
    public void setSystemRole(String role) {
        this.systemRole = role;
        this.chatMemory.clear();
        chatMemory.add(dev.langchain4j.data.message.SystemMessage.from(role));
    }

    /** Envoi d’un prompt au LLM + retour réponse */
    public String envoyer(String prompt) {
        return assistant.chat(prompt);
    }
}
