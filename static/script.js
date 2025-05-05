document.addEventListener('DOMContentLoaded', () => {
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const trainButton = document.getElementById('train-button');
    const trainingMessage = document.getElementById('training-message');
    const trainingStatusDiv = document.getElementById('training-status');

    // Function to add a message to the chatbox
    function addMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        // Use innerHTML to render potential HTML tags from the bot response
        messageDiv.innerHTML = text.replace(/\n/g, '<br>'); // Replace newlines with <br> for display
        chatbox.appendChild(messageDiv);
        // Scroll to the bottom
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    // Function to handle sending messages
    async function sendMessage() {
        const messageText = userInput.value.trim();
        if (messageText === '') return; // Avoid  sending empty messages

        // Add user message to chatbox
        addMessage('user', messageText);
        userInput.value = ''; // Clear input field

        // Add a temporary "Bot is thinking..."
        const thinkingMessage = document.createElement('div');
        thinkingMessage.classList.add('message', 'bot');
        thinkingMessage.textContent = 'Bot is thinking...';
        chatbox.appendChild(thinkingMessage);
        chatbox.scrollTop = chatbox.scrollHeight;


        try {
            // Send message to backend
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ user_message: messageText }),
            });

            // Remove the "thinking" message
            chatbox.removeChild(thinkingMessage);

            if (!response.ok) {
                // Handle HTTP errors (like 500 Internal Server Error)
                addMessage('bot', `Sorry, there was an error communicating with the server (Status: ${response.status}).`);
                return;
            }

            const data = await response.json();
            // Add bot response to chatbox
            addMessage('bot', data.response);

        } catch (error) {
             // Remove the "thinking" message even if there's an error
            if (chatbox.contains(thinkingMessage)) {
                chatbox.removeChild(thinkingMessage);
            }
            console.error("Error sending message:", error);
            addMessage('bot', 'Sorry, something went wrong. Please check the console or try again later.');
        }
    }

    // Function to handle training request
    async function handleTraining() {
        if (!trainButton) return; // Should not happen if button exists

        trainButton.disabled = true;
        trainButton.textContent = 'Training...';
        trainingMessage.textContent = 'Please wait, this might take a moment...';
        trainingMessage.style.color = '#007bff';

        try {
            const response = await fetch('/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json', // Though no body is sent, it's good practice
                },
            });

            const data = await response.json();

            if (data.status === 'success' || data.status === 'info') {
                trainingMessage.textContent = data.message;
                trainingMessage.style.color = 'green';
                // Update status visually and remove button
                trainingStatusDiv.innerHTML = ' <p><strong>Model Status:</strong> <span style="color: green; font-size: 18px; font-weight: bold;">Ready</span></p>';

                
            } else {
                trainingMessage.textContent = `Error: ${data.message}`;
                trainingMessage.style.color = 'red';
                trainButton.disabled = false; // Re-enable button on error
                trainButton.textContent = 'Train on e-commerce support Q&A data';
            }

        } catch (error) {
            console.error("Error triggering training:", error);
            trainingMessage.textContent = 'An error occurred while trying to train the model. Please check the console.';
            trainingMessage.style.color = 'red';
            trainButton.disabled = false; 
            trainButton.textContent = 'Train on e-commerce support Q&A data';
        }
    }

    // Event Listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        // Send message if Enter key is pressed
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    // Add event listener only if the train button exists
    if (trainButton) {
        trainButton.addEventListener('click', handleTraining);
    }
});