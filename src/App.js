import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import './App.css';
import { HashRouter, Route, Routes } from 'react-router';
import Login from './pages/auth/Login';
import Register from './pages/auth/Register';
import Home from './pages/Home';
import { Toaster } from './components/ui/toaster';
function App() {
    return (_jsxs(HashRouter, { children: [_jsx(Toaster, {}), _jsxs(Routes, { children: [_jsx(Route, { path: "/auth/login", element: _jsx(Login, {}) }), _jsx(Route, { path: "/auth/register", element: _jsx(Register, {}) }), _jsx(Route, { path: "/home/:userId/:titleId?", element: _jsx(Home, {}) })] })] }));
}
export default App;
