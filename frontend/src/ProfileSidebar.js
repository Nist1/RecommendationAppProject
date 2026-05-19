import { useState, useEffect } from 'react';
import axios from 'axios';
import './ProfileSidebar.css';

const api = axios.create({ baseURL: 'http://127.0.0.1:8000' });

api.interceptors.request.use(config => {
  const token = localStorage.getItem('authToken');
  if (token) config.headers['Authorization'] = `Token ${token}`;
  return config;
});

function ProfileSidebar({ isOpen, onClose, user, onUserChange, onQuerySelect }) {
  const [mode, setMode] = useState('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [history, setHistory] = useState([]);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    if (isOpen && user) {
      api.get('/api/auth/history/')
        .then(res => {
          if (res.data.status === 'success') {
            setHistory(res.data.history);
            setTotal(res.data.total);
          }
        })
        .catch(() => {});
    }
  }, [isOpen, user]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    try {
      const url = mode === 'login' ? '/api/auth/login/' : '/api/auth/register/';
      const res = await api.post(url, { username, password });
      if (res.data.status === 'success') {
        localStorage.setItem('authToken', res.data.token);
        onUserChange(res.data.username);
        setUsername('');
        setPassword('');
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Ошибка');
    }
  };

  const handleLogout = async () => {
    await api.post('/api/auth/logout/');
    localStorage.removeItem('authToken');
    onUserChange(null);
    setHistory([]);
    setTotal(0);
  };

  return (
    <>
      <div className={`sidebarOverlay ${isOpen ? 'sidebarOverlayVisible' : ''}`} onClick={onClose} />
      <div className={`sidebar ${isOpen ? 'sidebarOpen' : ''}`}>
        <button className='sidebarClose' onClick={onClose}>×</button>

        {user ? (
          <div className='sidebarProfile'>
            <div className='sidebarAvatar'>{user[0].toUpperCase()}</div>
            <h2 className='sidebarUsername'>{user}</h2>
            <div className='sidebarStat'>
              <span className='sidebarStatNumber'>{total}</span>
              <span className='sidebarStatLabel'>запросов всего</span>
            </div>
            <button className='sidebarLogout' onClick={handleLogout}>Выйти</button>

            {history.length > 0 && (
              <div className='sidebarHistory'>
                <div className='sidebarHistoryHeader'>
                  <h3 className='sidebarHistoryTitle'>История поиска</h3>
                  <button className='sidebarHistoryClear' onClick={async () => {
                    await api.post('/api/auth/history/clear/');
                    setHistory([]);
                    setTotal(0);
                  }}>Очистить</button>
                </div>
                <ul className='sidebarHistoryList'>
                  {history.map((item, i) => (
                    <li key={i} className='sidebarHistoryItem' onClick={() => { onQuerySelect(item.query); onClose(); }}>
                      <span className='sidebarHistoryQuery'>{item.query}</span>
                      <span className='sidebarHistoryTime'>{item.timestamp}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <div className='sidebarAuth'>
            <div className='sidebarModeToggle'>
              <span className={`sidebarModeLabel ${mode === 'login' ? 'sidebarModeLabelActive' : ''}`}>Вход</span>
              <label className='sidebarModeSlider'>
                <input
                  type='checkbox'
                  checked={mode === 'register'}
                  onChange={e => { setMode(e.target.checked ? 'register' : 'login'); setError(''); }}
                />
                <span className='sidebarModeTrack' />
              </label>
              <span className={`sidebarModeLabel ${mode === 'register' ? 'sidebarModeLabelActive' : ''}`}>Регистрация</span>
            </div>

            <form onSubmit={handleSubmit} className='sidebarForm'>
              <input
                className='sidebarInput'
                type='text'
                placeholder='Имя пользователя'
                value={username}
                onChange={e => setUsername(e.target.value)}
                autoComplete='username'
              />
              <input
                className='sidebarInput'
                type='password'
                placeholder='Пароль'
                value={password}
                onChange={e => setPassword(e.target.value)}
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              />
              {error && <p className='sidebarError'>{error}</p>}
              <button className='sidebarSubmit' type='submit'>
                {mode === 'login' ? 'Войти' : 'Зарегистрироваться'}
              </button>
            </form>
          </div>
        )}
      </div>
    </>
  );
}

export default ProfileSidebar;
