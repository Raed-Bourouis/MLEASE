import { useState } from 'react'

import HomePage from './pages/Homepage'
import AuthPage from './Pages/AuthPage'
function App() {
  const [count, setCount] = useState(0)

  return (
    <>
    <AuthPage />
    </>

  )
}

export default App
