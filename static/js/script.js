document.getElementById('qa-form').onsubmit = function(event) {
    event.preventDefault();
    var question = document.getElementById('question').value;
    addMessage(question, 'question');
    fetch('/api/doc_qa', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question }),
    })
    .then(response => response.json())
    .then(data => {
        addMessage(data.answer, 'answer');
    })
    .catch((error) => {
        console.error('Error:', error);
        addMessage('Sorry, an error occurred.', 'answer');
    });
    document.getElementById('question').value = ''; // Clear input after sending
};

function addMessage(text, className) {
    var chatBox = document.getElementById('chat-box');
    var messageDiv = document.createElement('div');
    var containerDiv = document.createElement('div'); // 创建一个新的div作为消息容器
    containerDiv.classList.add('container');
    
    // 根据消息类型（问题或回答），设置容器的类
    if (className === 'question') {
        containerDiv.classList.add('question-container');
    }

    messageDiv.classList.add('message', className);
    messageDiv.textContent = text;
    containerDiv.appendChild(messageDiv); // 将消息div添加到容器div中
    chatBox.appendChild(containerDiv); // 然后将容器div添加到聊天框中
    chatBox.scrollTop = chatBox.scrollHeight; // 滚动到底部
}
