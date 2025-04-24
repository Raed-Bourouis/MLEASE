const Header = () => {
    console.log("Rendering Header")
  
    const navLinks = ["Products", "Solutions", "Resources"]
  
    return (
      <header className="w-full py-4 px-8 bg-white bg-opacity-80 backdrop-blur-md shadow-sm fixed top-0 z-30">
        <div className="container mx-auto flex justify-between items-center">
          {/* Logo */}
          <div className="text-lg font-bold text-gray-800">
            <img src="https://via.placeholder.com/40x40.png?text=Logo" alt="Logo" className="inline-block w-8 h-8 mr-2" />
            MLEASE
          </div>
  
          {/* Navigation */}
          <nav className="hidden md:flex space-x-8 text-gray-700">
            {navLinks.map(link => (
              <a key={link} href="#" className="hover:text-blue-600 transition">
                {link}
              </a>
            ))}
          </nav>
  
          {/* Buttons */}
          <div className="flex space-x-4">
            <button className="px-4 py-2 text-gray-700 hover:text-blue-600 transition">
              Login
            </button>
            <button className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded transition">
              Get Started Free →
            </button>
          </div>
        </div>
      </header>
    )
  }
  
  export default Header
  