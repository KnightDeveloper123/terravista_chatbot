import { Heading } from '@chakra-ui/react'
import './App.css'
import { HashRouter, Route, Routes } from 'react-router'
import Login from './pages/auth/Login'
import Register from './pages/auth/Register'
import Home from './pages/Home'
import { Toaster } from './components/ui/toaster'

function App() {
  return (
    <HashRouter>
      <Toaster />
      <Routes>
        <Route path="/auth/login" element={<Login />} />
        <Route path="/auth/register" element={<Register />} />
        <Route path="/home/:userId/:titleId?" element={<Home />} />
      </Routes>
    </HashRouter>
  )
}

export default App
