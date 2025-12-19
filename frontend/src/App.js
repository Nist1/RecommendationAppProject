import { useState } from 'react';
import { IonIcon } from '@ionic/react';
import { attach, search, } from 'ionicons/icons';
import axios from 'axios';
import ResultCard from './ResultCard';
import './App.css';

//Временные семплы рекомендаций, потом они будут приходить с бэкенда
const sampleRecs = [
  {
    title: "Warm Result",
    text: "The coziest search result you'll find today. Perfect for chilly evenings."
  },
  {
    title: "Toasty Discovery",
    text: "Just like fresh bread from the oven, this result will warm your heart."
  },
  {
    title: "Sunny Finding",
    text: "Bright and cheerful, like a summer day in the middle of winter."
  },
  {
    title: "Final Warmth",
    text: "The perfect ending to your search - warm, comforting, and satisfying."
  }
];

const results = [];

function App() {
  const [isRecsRequested, setRecsRequested] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [searchQuery, setSeacrhQuery] = useState('');
  const [results, setResults] = useState([]);

  const uploadDataset = async () => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.csv,.json';

    fileInput.onchange = async (event) => {
      const file = event.target.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('dataset', file);

      try {
        const response = await axios.post('http://127.0.0.1:8000/api/upload/', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percentCompleted);
          },
        });

        if (response.data.status === 'success') {
          alert('Файл успешно загружен!');
        } else {
          alert('Ошибка при загрузке файла! Повторите снова.');
        }
      } catch (error) {
        console.error('Ошибка: ', error);
        alert('Ошибка при загрузке файла! Повторите снова.')
      } finally {
        setUploadProgress(0);
      }
    }

    fileInput.click();
  }
  
  const displayResult = async () => {
    if (!searchQuery.trim()) {
      alert('Пожалуйста, введите запрос для поиска.');
      return;
    }

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/search/', {
        query: searchQuery,
      });

      if (response.data.status === 'success') {
        setResults(response.data.results || []);
        setRecsRequested(true);
      } else {
        alert('Ошибка выполнения.' + (response.data.message || ''));
      }
    } catch (error) {
      console.error('Ошибка: ', error);
      alert('Произошла ошибка при выполнении поиска.')
    }
  }

  const clearResult = () => {
    setRecsRequested(false);
  }
  
  return (
    <>
      <main className='mainContainer'>

        <div className={`searchContainer ${isRecsRequested ? 'searchRaised' : ''}`}>
          <button className='uploadButton' onClick={uploadDataset}>
            <IonIcon icon={attach} style={{ fontSize: '24px', color: '#fff' }} />
            Загрузить датасет
          </button>

          {uploadProgress > 0 && (
            <div className='progressBarContainer'>
              <progress
                className='progressBar'
                value={uploadProgress}
                max="100"
              />
              <span>{uploadProgress}</span>
            </div>
          )}

          <div className='searchBar'>
            <input 
              type='text'
              className='searchInput' 
              placeholder='Введите запрос для поиска рекомендаций'
              value={searchQuery}
              onChange={(query) => setSeacrhQuery(query.target.value)}
            />
            <button className='searchButton' onClick={displayResult}>
              <IonIcon icon={search} style={{ fontSize: '24px', color: '#fff' }} />
            </button>
          </div>
        </div>

        <div className='resultsContainer' style={{ display: isRecsRequested ? 'flex' : 'none' }}>
          {isRecsRequested && results.map((rec, index) => (
            <ResultCard
              key={rec.id || index}
              title={rec.title}
              content={rec.text}
              index={index}
            />
          ))}
        </div>

        <div className='bottomButtonsContainer' style={{ display: isRecsRequested ? 'flex' : 'none' }}>
          <button className='clearButton' onClick={clearResult}>
            Очистить рекомендации
          </button>
        </div>

      </main>
    </>
  );
}

export default App;
