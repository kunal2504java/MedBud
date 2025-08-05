import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://localhost:8000', // Adjust this if your backend runs on a different port
  headers: {
    'Content-Type': 'application/json',
  },
});

export const register = (email, password) => {
  return apiClient.post('/users/register', { email, password });
};

export const login = (email, password) => {
  const formData = new FormData();
  formData.append('username', email);
  formData.append('password', password);

  return apiClient.post('/users/login', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

export default apiClient;
