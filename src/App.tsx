import './App.css'
import { HashRouter, Navigate, Route, Routes } from 'react-router'
import Login from './pages/auth/Login'
import Register from './pages/auth/Register'
import Home from './pages/Home'

import AdminLogin from './pages/auth/AdminLogin'

import { Toaster } from './components/ui/toaster'

function App() {
  return (
    <HashRouter>
      <Toaster />
      <Routes>
        <Route path="/" element={<Navigate to="/auth/login" replace />} />

        <Route path="/auth/login" element={<Login />} />
        <Route path="/auth/register" element={<Register />} />
        <Route path="/home/:userId/:titleId?" element={<Home />} />

        <Route path="/admin/login" element={<AdminLogin />} />

      </Routes>
    </HashRouter>
  )
}

export default App
