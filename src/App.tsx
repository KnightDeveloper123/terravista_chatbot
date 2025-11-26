import './App.css'
import { HashRouter, Navigate, Route, Routes } from 'react-router'
import Login from './pages/auth/Login'
import Register from './pages/auth/Register'
import Home from './pages/Home'

import AdminLogin from './pages/auth/AdminLogin'

import { Toaster } from './components/ui/toaster'
import AdminLayout from './pages/admin/AdminLayout'
import Dashboard from './pages/admin/Dashboard'
import Users from './pages/admin/Users'
import ScheduleMeeting from './pages/ScheduleMeeting'

function App() {
  return (
    <HashRouter>
      <Toaster />
      <Routes>
        <Route path="/" element={<Navigate to="/auth/login" replace />} />

        <Route path="/auth/login" element={<Login />} />
        <Route path="/auth/register" element={<Register />} />
        <Route path="/home/:userId/:titleId?" element={<Home />} />
        <Route path="/schedule-meeting/:userId" element={<ScheduleMeeting />} />

        <Route path="/admin/login" element={<AdminLogin />} />
        <Route path='/admin/*' element={<AdminLayout />}>
          <Route path='dashboard' element={<Dashboard />} />
          <Route path='users' element={<Users />} />
        </Route>

      </Routes>
    </HashRouter>
  )
}

export default App
