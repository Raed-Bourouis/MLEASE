import { ChevronRight, Code, BarChart2, RefreshCw, Shield } from "lucide-react";
import mlease_logo from "../../assets/logo.svg";

const Navbar = () => {
  return (
    <nav className="flex justify-around items-center bg-[#FD9D65] py-4 px-6 md:px-12 lg:px-30 h-24 ">
      <div className="flex flex-row items-center space-x-8 h-20">
        <img
          className="h-16 border- pt-1 "
          src={mlease_logo}
          alt=""
          width={130}
        />
        <div className="hidden md:flex space-x-8 p-0 m-0 text-[#344767] text-xl font-bold h-16 -2 items-center">
          <a
            href="#"
            className="hover:text-white/50 transition-colors duration-250"
          >
            Products
          </a>
          <a href="#" className="hover:text-white/50 transition-colors">
            Solutions
          </a>
          <a href="#" className="hover:text-white/50 transition-colors">
            Resources
          </a>
        </div>
      </div>
      <div className="flex items-center space-x-4 text-lg font-bold h-20 ">
        <a
          href="#"
          className="bg-yellow-100 text-blue-900 px-5 py-2 rounded-md hover:bg-yellow-200 transition-colors"
        >
          Login
        </a>
        <a
          href="#"
          className="bg-blue-500 text-white px-5 py-2 rounded-md hover:bg-blue-600 transition-colors flex items-center "
        >
          Get Started Free →
        </a>
      </div>
    </nav>
  );
};

export default Navbar;
