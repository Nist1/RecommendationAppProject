import { useState, useRef, useEffect } from 'react';
import { IonIcon } from '@ionic/react';
import { attach, search, close } from 'ionicons/icons';
import axios from 'axios';
import ResultCard from './ResultCard';
import './App.css';

const HISTORY_KEY = 'searchHistory';
const MAX_HISTORY = 5;

function saveQueryToHistory(query) {
  const trimmed = query.trim();
  if (!trimmed) return;
  const existing = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  const deduped = existing.filter(q => q !== trimmed);
  deduped.push(trimmed);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(deduped.slice(-MAX_HISTORY)));
}

function getHistory() {
  return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
}

function App() {
  const [isRecsRequested, setRecsRequested] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [searchQuery, setSeacrhQuery] = useState('');
  const [results, setResults] = useState([]);
  const [historyResults, setHistoryResults] = useState([]);
  const [showHistoryDropdown, setShowHistoryDropdown] = useState(false);
  const [activeHistoryIndex, setActiveHistoryIndex] = useState(-1);
  const [method, setMethod] = useState('tfidf');
  const searchContainerRef = useRef(null);

  const getFilteredHistory = () => {
    const history = getHistory();
    if (!searchQuery.trim()) return history.slice().reverse();
    return history.slice().reverse().filter(q =>
      q.toLowerCase().includes(searchQuery.toLowerCase())
    );
  };

  const removeFromHistory = (queryToRemove) => {
    const existing = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    localStorage.setItem(HISTORY_KEY, JSON.stringify(existing.filter(q => q !== queryToRemove)));
  };

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (searchContainerRef.current && !searchContainerRef.current.contains(e.target)) {
        setShowHistoryDropdown(false);
        setActiveHistoryIndex(-1);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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
            if (percentCompleted === 100) {
              setUploadProgress(0);
              setIsProcessing(true);
            }
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
        setIsProcessing(false);
      }
    }

    fileInput.click();
  }

  const displayResult = async () => {
    if (!searchQuery.trim()) {
      alert('Пожалуйста, введите запрос для поиска.');
      return;
    }

    saveQueryToHistory(searchQuery);
    const previousHistory = getHistory().slice(0, -1);

    setHistoryResults([]);
    setShowHistoryDropdown(false);
    setSeacrhQuery('');

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/search/', {
        query: searchQuery,
        method,
      });

      if (response.data.status === 'success') {
        const mainResults = response.data.results || [];
        setResults(mainResults);
        setRecsRequested(true);

        if (previousHistory.length > 0) {
          try {
            const histRes = await axios.post('http://127.0.0.1:8000/api/history-recs/', {
              history: previousHistory,
              exclude_ids: mainResults.map(r => r.id),
              method,
            });
            if (histRes.data.status === 'success') {
              setHistoryResults(histRes.data.results || []);
            }
          } catch (historyError) {
            console.error('Ошибка загрузки истории рекомендаций:', historyError);
          }
        }
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
    setHistoryResults([]);
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
              <span>{uploadProgress}%</span>
            </div>
          )}

          {isProcessing && (
            <div className='progressBarContainer'>
              <progress className='progressBar progressBarIndeterminate' />
              <span>Обработка датасета...</span>
            </div>
          )}

          <div className='methodToggle'>
            <span className={`methodLabel ${method === 'tfidf' ? 'methodLabelActive' : ''}`}>TF-IDF</span>
            <label className='methodSlider'>
              <input
                type='checkbox'
                checked={method === 'sbert'}
                onChange={(e) => setMethod(e.target.checked ? 'sbert' : 'tfidf')}
              />
              <span className='methodSliderTrack' />
            </label>
            <span className={`methodLabel ${method === 'sbert' ? 'methodLabelActive' : ''}`}>Sentence Transformers</span>
          </div>

          <div className='searchBar' ref={searchContainerRef}>
            <input
              type='text'
              className='searchInput'
              placeholder='Введите запрос для поиска рекомендаций'
              value={searchQuery}
              onChange={(e) => {
                setSeacrhQuery(e.target.value);
                setShowHistoryDropdown(true);
                setActiveHistoryIndex(-1);
              }}
              onFocus={() => setShowHistoryDropdown(true)}
              onKeyDown={(e) => {
                const filtered = getFilteredHistory();
                if (!showHistoryDropdown || filtered.length === 0) {
                  if (e.key === 'Enter') displayResult();
                  return;
                }
                if (e.key === 'ArrowDown') {
                  e.preventDefault();
                  setActiveHistoryIndex(i => Math.min(i + 1, filtered.length - 1));
                } else if (e.key === 'ArrowUp') {
                  e.preventDefault();
                  setActiveHistoryIndex(i => Math.max(i - 1, -1));
                } else if (e.key === 'Enter') {
                  if (activeHistoryIndex >= 0) {
                    setSeacrhQuery(filtered[activeHistoryIndex]);
                    setShowHistoryDropdown(false);
                    setActiveHistoryIndex(-1);
                  } else {
                    displayResult();
                  }
                } else if (e.key === 'Escape') {
                  setShowHistoryDropdown(false);
                  setActiveHistoryIndex(-1);
                }
              }}
            />
            {showHistoryDropdown && getFilteredHistory().length > 0 && (
              <ul className='historyDropdown'>
                {getFilteredHistory().map((item, index) => (
                  <li
                    key={item}
                    className={`historyDropdownItem ${index === activeHistoryIndex ? 'historyDropdownItemActive' : ''}`}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      setSeacrhQuery(item);
                      setShowHistoryDropdown(false);
                      setActiveHistoryIndex(-1);
                    }}
                    onMouseEnter={() => setActiveHistoryIndex(index)}
                  >
                    <IonIcon icon={search} className='historyDropdownIcon' />
                    <span className='historyDropdownText'>{item}</span>
                    <button
                      className='historyDropdownRemove'
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        removeFromHistory(item);
                        setActiveHistoryIndex(-1);
                      }}
                    >
                      <IonIcon icon={close} />
                    </button>
                  </li>
                ))}
              </ul>
            )}
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

        {historyResults.length > 0 && (
          <div className='historySection'>
            <h2 className='historySectionTitle'>Вам также может быть интересно</h2>
            <div className='historyResultsContainer'>
              {historyResults.map((rec, index) => (
                <ResultCard
                  key={rec.id}
                  title={rec.title}
                  content={rec.text}
                  index={index}
                />
              ))}
            </div>
          </div>
        )}

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
