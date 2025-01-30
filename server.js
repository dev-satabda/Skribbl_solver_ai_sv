require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const { ChatGoogleGenerativeAI } = require('@langchain/google-genai');
const { ChatPromptTemplate } = require('@langchain/core/prompts');

const app = express();
const port = process.env.PORT || 5000;
app.use(bodyParser.json({ limit: '10mb' }));

// Initialize Gemini AI with API key from .env
const llm = new ChatGoogleGenerativeAI({
  model: 'gemini-1.5-flash',
  apiKey: process.env.GEMINI_API_KEY,
  temperature: 0,
  maxRetries: 2,
});

const prompt = ChatPromptTemplate.fromMessages([
  [
    'system',
    `You're an AI that helps solve Skribbl.io by analyzing a drawing. 
    Based on the drawing and possible hints (e.g., "_ _ _ _ _ (5)" means a 5-letter word), predict the 10 most likely words.
    
    🔹 **Strict Output Format:**  
    Return an array of words in **valid JSON format** like this:
    
    ["word1", "word2", "word3", ..., "word10"]
    
    🔹 **Important Rules:**  
    - Only return an **array** (no extra text, explanations, or numbering).
    - Each item in the array should be a **single word** (no phrases).
    - Ensure proper JSON syntax.
    `,
  ],
  ['human', '{image}'],
]);

const chain = prompt.pipe(llm);

// Retry logic with exponential backoff
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchPrediction(image) {
  let attempts = 3;
  for (let i = 0; i < attempts; i++) {
    try {
      console.log(`Attempt ${i + 1}: Sending image to Gemini AI for analysis...`);
      
      const aiResponse = await chain.invoke({ image });
      let responseText = aiResponse.content.trim();

      // Extract JSON from possible backticks
      const jsonMatch = responseText.match(/```json\n([\s\S]*?)\n```/);
      if (jsonMatch) {
        responseText = jsonMatch[1]; // Extract content between backticks
      }

      // Ensure valid JSON format
      const words = JSON.parse(responseText);
      if (Array.isArray(words)) {
        return words;
      } else {
        throw new Error("AI response is not a valid JSON array.");
      }
      
    } catch (error) {
      console.error(`Attempt ${i + 1} failed:`, error.message);
      if (i < attempts - 1) {
        await delay(2000 * (i + 1));
      } else {
        throw error;
      }
    }
  }
}


app.post('/upload', async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) return res.status(400).json({ error: 'No image provided' });

    const words = await fetchPrediction(image);
    res.json({ words });
  } catch (error) {
    console.error('Error:', error);
    res.status(503).json({ error: 'Gemini AI is unavailable. Try again later.' });
  }
});

app.listen(port, () => console.log(`Server running on http://localhost:${port}`));
