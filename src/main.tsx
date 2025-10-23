import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import { Provider } from './components/ui/provider'
import AppProviders from './context/AppProvider'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Provider>
      <AppProviders>
        <App />
      </AppProviders>
    </Provider>
  </StrictMode>,
)
