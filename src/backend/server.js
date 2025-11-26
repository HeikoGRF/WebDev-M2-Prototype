
import express from 'express';
import cors from 'cors';
import { pipeline } from '@xenova/transformers';

const app = express();
const port = 3001;

app.use(cors());
app.use(express.json());

// Global variable to hold the pipeline
let extractor = null;

// Initialize the pipeline
async function initPipeline() {
  try {
    console.log('Loading model...');
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Failed to load model:', error);
  }
}

// Initialize on startup
initPipeline();

app.post('/embed', async (req, res) => {
  try {
    const { text } = req.body;
    
    if (!text) {
      return res.status(400).json({ error: 'Text is required' });
    }

    if (!extractor) {
      // Try to initialize again if it failed or hasn't finished
      await initPipeline();
      if (!extractor) {
        return res.status(500).json({ error: 'Model not initialized' });
      }
    }

    const output = await extractor(text, { pooling: 'mean', normalize: true });
    const embedding = Array.from(output.data);

    res.json({ embedding });
  } catch (error) {
    console.error('Error processing request:', error);
    res.status(500).json({ error: 'Failed to generate embedding' });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

