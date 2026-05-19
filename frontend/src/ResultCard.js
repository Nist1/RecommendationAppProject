import { useEffect, useState } from 'react';
import axios from 'axios';
import './ResultCard.css';

function ModalCard({
  isOpen,
  onClose,
  title,
  content,
  similarItems,
  isLoadingSimilar,
  similarError,
}) {
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className='modalOverlay' onClick={onClose}>
      <div className='modalCard' onClick={(e) => e.stopPropagation()}>
        <button className='closeButton' onClick={onClose}>x</button>
        <h2>{title}</h2>
        <p>{content}</p>

        <div className='modalSimilarBlock'>
          <h3 className='similarTitle'>Похожие статьи</h3>

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
      </div>
    </div>
  );
}

function ResultCard({ id, title, content, index }) {
  const [isModalOpen, setIsModalOpen] = useState(false);
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

  const handleOpen = () => {
    setIsModalOpen(true);
    loadSimilarItems();
  };

  const handleClose = () => {
    setIsModalOpen(false);
  };

  return (
    <>
      <div
        className='resultCard'
        style={{ animationDelay: `${index * 0.1}s` }}
        onClick={handleOpen}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            handleOpen();
          }
        }}
      >
        <h3 className='cardTitle'>{title}</h3>
        <p className='cardContent'>{content}</p>
      </div>

      <ModalCard
        isOpen={isModalOpen}
        onClose={handleClose}
        title={title}
        content={content}
        similarItems={similarItems}
        isLoadingSimilar={isLoadingSimilar}
        similarError={similarError}
      />
    </>
  );
}

export default ResultCard;
