import './ResultCard.css';
import { useState } from 'react';

function ModalCard({isOpen, onClose, title, content}) {
  if (!isOpen) return null;

  return (
    <div className='modalOverlay'>
      <div className='modalCard'>
        <button className='closeButton' onClick={onClose}>x</button>
        <h2>{title}</h2>
        <p>{content}</p>
      </div>
    </div>
  )
}

function ResultCard({title, content, index}) {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleOpen = () => {
    setIsModalOpen(true);
  }

  const handleClose = () => {
    setIsModalOpen(false);
  }

  return (
    <>
      <div 
        className='resultCard' 
        style={{ animationDelay: `${index * 0.1}s` }}
        onClick={handleOpen}
      >
        <h3 className='cardTitle'>{title}</h3>
        <p className='cardContent'>{content}</p>
      </div>
      
      <ModalCard 
        isOpen={isModalOpen}
        onClose={handleClose}
        title={title}
        content={content}
      />
    </>
  )
}

export default ResultCard;
