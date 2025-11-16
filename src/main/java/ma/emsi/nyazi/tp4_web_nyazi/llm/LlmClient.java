package ma.emsi.nyazi.tp4_web_nyazi.llm;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;
import jakarta.enterprise.context.Dependent;

import java.io.Serializable;


@Dependent
public class LlmClient implements Serializable {
    public interface Assistant {
        String chat(String prompt);
    }

    private String systemRole;
    private final ChatMemory chatMemory;
    private final Assistant assistant;

    public LlmClient() {

        String apiKey = System.getenv("GEMINI_KEY");  // ✅ même clé que TP1
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .build();

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
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
