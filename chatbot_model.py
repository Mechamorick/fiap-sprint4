import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Perguntas e respostas comuns
perguntas = [
    "Oi", "Olá", "Tudo bem?", "Como você está?", "Qual o seu nome?", "O que você faz?",
    "Ajuda", "Preciso de suporte", "Consultas comerciais", "Informações sobre medicamentos",
    "Qual é o horário de funcionamento?", "Como posso entrar em contato?", "Você pode me ajudar?",
    "O que é o seu serviço?", "Você tem um e-mail?"
]

respostas = [
    "Olá!", "Oi!", "Estou bem, obrigado!", "Estou ótimo, e você?", "Eu sou um chatbot.", "Eu ajudo você com suas dúvidas.",
    "Claro, em que posso ajudar?", "Suporte disponível, qual o problema?", "Você está no setor comercial, em que posso ajudar?",
    "Aqui estão as informações sobre medicamentos.", "Estamos abertos das 9h às 18h.", "Você pode entrar em contato pelo telefone ou e-mail.",
    "Sim, estou aqui para ajudar!", "Oferecemos serviços de consulta e suporte ao cliente.", "Nosso e-mail é contato@exemplo.com."
]

# Codificador de respostas
encoder = LabelEncoder()
encoded_respostas = encoder.fit_transform(respostas)

# Tokenização das perguntas
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(perguntas)
sequencias_perguntas = tokenizer.texts_to_sequences(perguntas)

# Preencher as sequências para que todas tenham o mesmo tamanho
max_len = 10  # Ajuste do max_len para um valor adequado
sequencias_perguntas = tf.keras.preprocessing.sequence.pad_sequences(sequencias_perguntas, maxlen=max_len)

# Ajustando o input_dim da camada Embedding ao tamanho do vocabulário
vocab_size = len(tokenizer.word_index) + 1  # +1 para incluir o zero no index

# Dados prontos
num_classes = len(set(encoded_respostas))

# Treinar o modelo
def treinar_modelo():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        tf.keras.layers.Flatten(),  # Usando Flatten para achatar a saída
        tf.keras.layers.Dense(64, activation='relu'),  # Ajuste para 64 unidades
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Camada de saída
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(sequencias_perguntas, np.array(encoded_respostas), epochs=200, verbose=1)
    model.save('chatbot_model.h5')
    print("Modelo treinado e salvo como 'chatbot_model.h5'.")

# Função para carregar o modelo treinado
def carregar_modelo():
    return tf.keras.models.load_model('chatbot_model.h5')

# Inicializar o Flask
app = Flask(__name__)

# Verifica se o modelo já foi treinado
try:
    model = carregar_modelo()
except Exception as e:
    print("Modelo não encontrado. Iniciando treinamento...")
    treinar_modelo()
    model = carregar_modelo()

# Inicializar o histórico da conversa
historico_conversa = []  # Inicializamos a variável global aqui

# Função para pré-processar a mensagem do usuário
def preprocess_message(message):
    # Converter a mensagem para minúsculas e remover espaços
    message = message.lower().strip()
    print(f"Mensagem pré-processada: {message}")
    return message

# Função para prever resposta com o modelo treinado
def predict_response(message):
    try:
        # Pré-processar a mensagem
        message = preprocess_message(message)
        
        # Converter a mensagem do usuário em sequência
        sequencia = tokenizer.texts_to_sequences([message])
        
        # Imprimir a sequência tokenizada
        print(f"Sequência tokenizada: {sequencia}")
        
        # Preencher a sequência para ter o mesmo tamanho
        sequencia = tf.keras.preprocessing.sequence.pad_sequences(sequencia, maxlen=max_len)

        # Verifique o tamanho da sequência
        print("Tamanho da sequência:", sequencia.shape)  # Debug

        # Fazer a previsão
        predicao = model.predict(sequencia)
        print("Predição:", predicao)  # Debug para a previsão
        classe_predita = np.argmax(predicao, axis=1)[0]
        resposta = encoder.inverse_transform([classe_predita])[0]
        return resposta
    except Exception as e:
        print("Erro ao prever resposta:", e)
        return "Desculpe, não entendi sua mensagem."

# Rota para previsões
@app.route('/predict', methods=['POST'])
def predict():
    global historico_conversa  # Certifique-se de que estamos acessando a variável global
    try:
        data = request.json
        message = data.get('message')

        # Adicionar a nova mensagem ao histórico
        historico_conversa.append({"user": message})
        
        # Obter a resposta do chatbot
        response = predict_response(message)
        
        # Adicionar a resposta do chatbot ao histórico
        historico_conversa.append({"bot": response})
        
        return jsonify({'response': response, 'historico': historico_conversa})
    except Exception as e:
        print("Erro ao processar a requisição:", e)
        return jsonify({'error': 'Erro interno do servidor.'}), 500

if __name__ == '__main__':
    app.run(port=5000)
