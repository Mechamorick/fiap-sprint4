const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const path = require('path'); // Importa o módulo path
const app = express();
const port = 3030;

// Middleware para analisar o corpo das requisições em JSON
app.use(bodyParser.json());
app.use(express.static('public')); // Para servir arquivos estáticos

// Rota para servir o index.html na raiz
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname,'index.html')); // Certifique-se de que index.html está na pasta public
});

// Endpoint para lidar com previsões
app.post('/chatbot', async (req, res) => {
    const message = req.body.message;

    try {
        // Requisição ao backend do Flask
        const response = await axios.post('http://localhost:5000/predict', { message });

        // Retorna a resposta ao frontend
        res.json({ response: response.data.response });
    } catch (error) {
        console.error('Erro ao se comunicar com o backend Flask:', error);
        res.status(500).json({ response: 'Erro ao se comunicar com o backend Flask' });
    }
});

// Inicie o servidor
app.listen(port, () => {
    console.log(`Servidor rodando em http://localhost:${port}`);
});
