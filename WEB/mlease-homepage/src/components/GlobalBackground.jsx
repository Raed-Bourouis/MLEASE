const GlobalBackground = () => (
    <>
      {/* Optional overlay for better text readability */}
      <div className="fixed inset-0 -z-10 bg-white/50" />
      
      {/* Main SVG background */}
      <div
        className="fixed inset-0 -z-20 bg-repeat bg-cover bg-fixed"
        style={{ backgroundImage: "url('/src/assets/background-pattern.svg')" }}
      />
    </>
  )
  
  export default GlobalBackground
  