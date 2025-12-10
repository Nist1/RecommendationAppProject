import './ResultCard.css';

function ResultCard({title, content, index}) {
  return (
    <div className='resultCard' style={{ animationDelay: `${index * 0.1}s` }}>
      <h3 className='cardTitle'>{title}</h3>
      <p className='cardContent'>{content}</p>
    </div>
  )
}

export default ResultCard;
