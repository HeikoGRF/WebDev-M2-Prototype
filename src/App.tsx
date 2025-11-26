import { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { PCA } from 'ml-pca';
import './App.css';

interface EmbeddingData {
  text: string;
  vector: number[];
}

function App() {
  const [text, setText] = useState('');
  const [embeddings, setEmbeddings] = useState<EmbeddingData[]>([]);
  const [loading, setLoading] = useState(false);

  const handleGenerateEmbedding = async () => {
    if (!text) return;
    
    try {
      setLoading(true);
      const response = await fetch('http://localhost:3001/embed', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }

      const data = await response.json();
      setEmbeddings(prev => [...prev, { text, vector: data.embedding }]);
      setText(''); // Clear input after success
    } catch (error) {
      console.error('Error generating embedding:', error);
      alert(`Failed to generate embedding: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setLoading(false);
    }
  };

  // Compute PCA coordinates
  const plotData = useMemo(() => {
    if (embeddings.length === 0) return { x: [], y: [], text: [] };

    const vectors = embeddings.map(e => e.vector);
    const texts = embeddings.map(e => e.text);

    let x: number[] = [];
    let y: number[] = [];

    if (embeddings.length === 1) {
      x = [0];
      y = [0];
    } else {
      try {
        // Reduce to 2 dimensions
        const pca = new PCA(vectors);
        const predict = pca.predict(vectors, { nComponents: 2 });
        
        // predict is a Matrix, we need to convert it
        for (let i = 0; i < predict.rows; i++) {
          x.push(predict.get(i, 0));
          y.push(predict.get(i, 1));
        }
      } catch (e) {
        console.error("PCA Calculation failed", e);
        // Fallback for edge cases or very small datasets where PCA might fail
        x = vectors.map((_, i) => i);
        y = vectors.map(() => 0);
      }
    }

    return { x, y, text: texts };
  }, [embeddings]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Perplexify</h1>
      </header>
      <div className="App-body">
        <div className="input-container">
          <input 
            type="text" 
            className="text-input" 
            placeholder="Enter text here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleGenerateEmbedding()}
          />
          <button 
            className="add-embedding-button"
            onClick={handleGenerateEmbedding}
            disabled={loading}
          >
            {loading ? 'Generating...' : 'add embedding'}
          </button>
        </div>
        
        <div className="chart-container">
          <Plot
            data={[
              {
                x: plotData.x,
                y: plotData.y,
                text: plotData.text,
                mode: 'text+markers',
                type: 'scatter',
                textposition: 'top center',
                marker: { size: 12, color: '#4a90e2' },
              },
            ]}
            layout={{ 
              autosize: true, 
              title: { text: 'Semantic Similarity (PCA Projection)' },
              hovermode: 'closest',
              xaxis: { title: { text: 'PC1' }, showgrid: true, zeroline: true },
              yaxis: { title: { text: 'PC2' }, showgrid: true, zeroline: true },
              margin: { l: 50, r: 50, b: 50, t: 50 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
            }}
            useResizeHandler={true}
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </div>
    </div>
  );
}

export default App;

