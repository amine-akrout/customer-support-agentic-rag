document.getElementById("chat-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const questionInput = document.getElementById("question");
    const responseContainer = document.getElementById("response");

    const question = questionInput.value.trim();
    if (!question) return;

    // Clear input
    questionInput.value = "";

    // Add user's question to the response container
    const userMessage = document.createElement("p");
    userMessage.textContent = `You: ${question}`;
    userMessage.style.color = "#1976d2"; // Blue for user messages
    responseContainer.appendChild(userMessage);

    // Add a loading spinner
    const loadingMessage = document.createElement("p");
    loadingMessage.textContent = "Bot: Thinking...";
    loadingMessage.style.color = "#888";
    responseContainer.appendChild(loadingMessage);

    try {
        // Send question to the FastAPI backend
        const response = await fetch("/answer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question }),
        });

        if (!response.ok) {
            throw new Error("Failed to get a response from the chatbot.");
        }

        const data = await response.json();

        // Remove the loading message
        responseContainer.removeChild(loadingMessage);

        // Add chatbot's response to the response container
        const botMessage = document.createElement("p");
        botMessage.textContent = `Bot: ${data.llm_output || data.message}`;
        botMessage.style.color = "#333";
        responseContainer.appendChild(botMessage);
    } catch (error) {
        console.error(error);

        // Remove the loading message
        responseContainer.removeChild(loadingMessage);

        // Display error message
        const errorMessage = document.createElement("p");
        errorMessage.textContent = "Bot: Sorry, I'm having trouble answering your question. Please try again later.";
        errorMessage.style.color = "#ff0000";
        responseContainer.appendChild(errorMessage);
    }

    // Scroll to the bottom of the response container
    responseContainer.scrollTop = responseContainer.scrollHeight;
});