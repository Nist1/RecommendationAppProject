import { useState } from 'react';
import axios from 'axios';
import './ResultCard.css';

function ResultCard({ id, title, content, index }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [similarItems, setSimilarItems] = useState([]);
  const [isLoadingSimilar, setIsLoadingSimilar] = useState(false);
  const [similarError, setSimilarError] = useState('');

  const loadSimilarItems = async () => {
    if (id === undefined || id === null || similarItems.length > 0 || isLoadingSimilar) {
      return;
    }

    setIsLoadingSimilar(true);
    setSimilarError('');

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/similar/', {
        item_id: id,
        n: 3,
      });

      if (response.data.status === 'success') {
        setSimilarItems(response.data.results || []);
      } else {
        setSimilarError(response.data.message || 'Не удалось загрузить похожие статьи.');
      }
    } catch (error) {
      console.error('Ошибка загрузки похожих статей:', error);
      setSimilarError('Не удалось загрузить похожие статьи.');
    } finally {
      setIsLoadingSimilar(false);
    }
  };

  const handleClick = () => {
    const nextExpanded = !isExpanded;
    setIsExpanded(nextExpanded);

    if (nextExpanded) {
      loadSimilarItems();
    }
  };

  return (
    <div
      className={`resultCard ${isExpanded ? 'resultCardExpanded' : ''}`}
      style={{ animationDelay: `${index * 0.1}s` }}
      onClick={handleClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleClick();
        }
      }}
    >
      <h3 className='cardTitle'>{title}</h3>
      <p className='cardContent'>{content}</p>

      {isExpanded && (
        <div className='similarBlock'>
          <h4 className='similarTitle'>Похожие статьи</h4>

          {isLoadingSimilar && <p className='similarStatus'>Загрузка...</p>}
          {similarError && <p className='similarStatus similarError'>{similarError}</p>}

          {!isLoadingSimilar && !similarError && similarItems.length === 0 && (
            <p className='similarStatus'>Похожие статьи не найдены.</p>
          )}

          {!isLoadingSimilar && !similarError && similarItems.length > 0 && (
            <ul className='similarList'>
              {similarItems.map((item) => (
                <li className='similarItem' key={item.id}>
                  <span className='similarItemTitle'>{item.title}</span>
                  <span className='similarItemText'>{item.text}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

export default ResultCard;
