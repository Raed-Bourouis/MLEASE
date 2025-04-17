import GlobalBackground from './components/GlobalBackground'
import Header from './components/Header'
import HeroSection from './components/HeroSection'
import ProjectObjectives from './components/ProjectObjectives'

const App = () => {
  console.log("Rendering App")

  return (
    <div className="relative">
      <GlobalBackground />
      <Header />
      <main className="pt-24">
        <HeroSection />
        <ProjectObjectives />
      </main>
    </div>
  )
}

export default App
