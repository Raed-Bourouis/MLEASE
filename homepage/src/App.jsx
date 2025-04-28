import { useState } from 'react'
// import HomePage from './components/HomePage/HomePage.jsx'

import HomePage from './components/Homepage/Homepage'
function App() {
  const [count, setCount] = useState(0)

  return (
    <>
    <HomePage />
    </>

  )
}

export default App
