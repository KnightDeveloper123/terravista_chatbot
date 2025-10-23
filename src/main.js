import { jsx as _jsx } from "react/jsx-runtime";
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';
import { Provider } from './components/ui/provider';
import AppProviders from './context/AppProvider';
createRoot(document.getElementById('root')).render(_jsx(StrictMode, { children: _jsx(AppProviders, { children: _jsx(Provider, { children: _jsx(App, {}) }) }) }));
