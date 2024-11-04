import express from "express";
import axios from "axios";

const app = express();
app.use(express.json());

app.post("/ia", async (req, res) => {
  const { text } = req.body;
  try {
    const response = await axios.post("http://localhost:11434/api/generate", {
      model: "assistente",
      prompt: text,
      stream: false,
    });
    const respData = response.data.response;
    res.send(respData);    
  } catch (error) {
    res.status(500).send({ error: "Erro ao conectar com o modelo de IA" });
  }
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
