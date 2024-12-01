import React, { useState } from 'react';
import './App.css';
import Chat from './components/chat';
function App() {
  const ACCEPTABLE_RANGES = {
    snoring_rate: [0, 100],
    respiration_rate: [10, 40],
    body_temperature: [95.0, 104.0],
    limb_movement: [0, 50],
    blood_oxygen: [80, 100],
    eye_movement: [0, 30],
    sleeping_hours: [0, 24],
    heart_rate: [40, 180],
  };

  const [formData, setFormData] = useState({
    snoring_rate: '',
    respiration_rate: '',
    body_temperature: '',
    limb_movement: '',
    blood_oxygen: '',
    eye_movement: '',
    sleeping_hours: '',
    heart_rate: '',
  });
  const [response, setResponse] = useState('');
  const apikey="AIzaSyD9zU2jqbpQ-BXdg2Q180v6jjeGN0iboKw"

  const [remedy,setRemedy]=useState("")
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResponse('');
    try {
      const res = await fetch('http://127.0.0.1:8000/predict/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      if (!res.ok) {
        throw new Error('Failed to fetch data from the server');
      }
      const data = await res.json();
      await setResponse(data.predicted_stress_level);
      await setRemedy(data.remedy)
      console.log(response)


    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      {/* Navbar */}
      <header className="navbar">
        <h1>Stresswise Predictor</h1>
      </header>

      <form onSubmit={handleSubmit} className="form-container">
        {Object.keys(formData).map((key) => (
          <div key={key} className="form-group">
            <label htmlFor={key}>
              {key.replace('_', ' ')} (Range: {ACCEPTABLE_RANGES[key][0]} - {ACCEPTABLE_RANGES[key][1]}):
            </label>
            <input
              type="number"
              id={key}
              name={key}
              value={formData[key]}
              onChange={handleChange}
              step="any"
              min={ACCEPTABLE_RANGES[key][0]}
              max={ACCEPTABLE_RANGES[key][1]}
              required
              className="input-field"
            />
          </div>
        ))}
        <button type="submit" disabled={loading} className="submit-button">
          {loading ? 'Predicting...' : 'Submit'}
        </button>
      </form>
      {/* <Chat></Chat> */}

      {response && <Chat stress={response} remedy={remedy} apiKey={apikey}/>}
      {error && <div className="error">Error: {error}</div>}
    </div>
  );
}

export default App;
