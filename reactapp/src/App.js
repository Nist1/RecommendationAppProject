import { useState } from 'react';
import { IonIcon } from '@ionic/react';
import { attach, search, } from 'ionicons/icons'; 
import ResultCard from './ResultCard';
import './App.css';

//Временные семплы рекомендаций, потом они будут приходить с бэкенда
const sampleRecs = [
  {
    title: "Warm Result",
    content: "The coziest search result you'll find today. Perfect for chilly evenings."
  },
  {
    title: "Toasty Discovery",
    content: "Just like fresh bread from the oven, this result will warm your heart."
  },
  {
    title: "Sunny Finding",
    content: "Bright and cheerful, like a summer day in the middle of winter."
  },
  {
    title: "Final Warmth",
    content: "The perfect ending to your search - warm, comforting, and satisfying."
  }
];

function App() {
  const [isRecsRequested, setRecsRequested] = useState(false);

  const uploadDataset = () => {
    // TODO: Сделать логику загрузки файла и передачи на бэкенд
    alert("Файл загружен. Идет обработка...");
  }

  const displayResult = () => {
    setRecsRequested(true);
  }

  const clearResult = () => {
    setRecsRequested(false);
  }

  const searchMoreRecommendations = () => {
    // TODO: Добавить возможность получить еще рекомендаций
    alert("Запрошено больше рекомендаций")
  }

  return (
    <>
      <main className='mainContainer'>

        <div className={`searchContainer ${isRecsRequested ? 'searchRaised' : ''}`}>
          <button className='uploadButton' onClick={uploadDataset}>
            <IonIcon icon={attach} style={{ fontSize: '24px', color: '#fff' }} />
            Загрузить датасет</button>
          <div className='searchBar'>
            <input type='text' className='searchInput' placeholder='Введите запрос для поиска рекомендаций'></input>
            <button className='searchButton' onClick={displayResult}>
              <IonIcon icon={search} style={{ fontSize: '24px', color: '#fff' }} />
            </button>
          </div>
        </div>

        <div className='resultsContainer' style={{ display: isRecsRequested ? 'flex' : 'none' }}>
          {isRecsRequested && sampleRecs.map((rec, index) => (
            <ResultCard key={index} title={rec.title} content={rec.content} index={index} />
          ))}
        </div>

        <div className='bottomButtonsContainer' style={{ display: isRecsRequested ? 'flex' : 'none' }}>
          <button className='clearButton' onClick={clearResult}>
            Очистить рекомендации
          </button>
          <button className='moreButton' onClick={searchMoreRecommendations}>
            Больше...
          </button>
        </div>

      </main>
    </>
  );
}

export default App;
