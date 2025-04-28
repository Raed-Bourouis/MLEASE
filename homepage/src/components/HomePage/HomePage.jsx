import { ChevronRight, Code, BarChart2, RefreshCw, Shield } from "lucide-react";

import { useState } from "react";
import Navbar from "./Navbar";
import AIphoto from "../../assets/background_ai.png";

const HomePage = () => {
  return (
    <>
      <Navbar />
      <MLeaseLanding />
    </>
  );
};

export default HomePage;

function MLeaseLanding() {
  return (
    <div className="bg-linear-135 from-[#FD9D65] to-[#FFB88C] w-full min-h-screen flex items-center justify-center p-6">
      <div className="max-w-6xl w-full flex flex-col md:flex-row items-center justify-between gap-8">
        <div className="space-y-6 max-w-lg">
          <h1 className="text-5xl font-bold text-white drop-shadow-md">
            Getting Started with MLEASE!
          </h1>
          <p className="text-xl text-white">
            MLEASE is an MLOps-driven platform that simplifies and democratizes
            machine learning model deployment.
          </p>
          <button className="bg-blue-500 hover:bg-blue-600 text-white py-3 px-6 rounded-md font-medium transition-colors flex items-center">
            Get Started Free
          </button>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-lg max-w-sm">
          <img src={AIphoto} alt="" />
        </div>
      </div>
    </div>
  );
}

function MLeasePage() {
  const [showScrollButton, setShowScrollButton] = useState(false);

  // Show scroll button when user scrolls down
  if (typeof window !== "undefined") {
    window.addEventListener("scroll", () => {
      if (window.scrollY > 300) {
        setShowScrollButton(true);
      } else {
        setShowScrollButton(false);
      }
    });
  }

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-white font-sans">
      {/* Top Navigation Bar */}

      {/* Hero Section */}
      <section className="bg-orange-300 py-12 md:py-24 px-6 md:px-12 lg:px-24">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-6 drop-shadow-sm">
              Getting Started with MLEASE!
            </h1>
            <p className="text-lg text-gray-700 mb-8">
              MLEASE is an MLOps-driven platform that simplifies and
              democratizes machine learning model deployment.
            </p>
            <a
              href="#"
              className="inline-block bg-blue-500 text-white px-6 py-3 rounded-md hover:bg-blue-600 transition-colors flex items-center"
            >
              Get Started Free <ChevronRight className="w-4 h-4 ml-1" />
            </a>
          </div>
          <div className="flex justify-center">
            <div className="bg-white p-6 rounded-lg shadow-lg">
              <div className="text-orange-500">
                <svg viewBox="0 0 200 200" fill="none" className="w-64 h-64">
                  <path
                    d="M100 40C80 40 70 50 70 70C70 90 80 100 100 100C120 100 130 90 130 70C130 50 120 40 100 40Z"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    d="M60 160C60 140 80 120 100 120C120 120 140 140 140 160"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <circle cx="100" cy="85" r="10" fill="currentColor" />
                  <path
                    d="M85 110C85 105 95 105 95 105H105C105 105 115 105 115 110"
                    stroke="currentColor"
                    strokeWidth="3"
                  />
                  <path
                    d="M40 70L20 70M160 70L180 70M40 100L10 100M160 100L190 100M40 130L20 130M160 130L180 130"
                    stroke="currentColor"
                    strokeWidth="3"
                  />
                  <rect
                    x="80"
                    y="45"
                    width="40"
                    height="30"
                    rx="5"
                    stroke="currentColor"
                    strokeWidth="3"
                  />
                  <text
                    x="100"
                    y="65"
                    textAnchor="middle"
                    fill="currentColor"
                    fontSize="16"
                  >
                    AI
                  </text>
                </svg>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Second Section */}
      <section className="py-16 px-6 md:px-12 lg:px-24">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div>
            <h2 className="text-3xl font-bold text-gray-800 mb-6">
              Simplify AI Operations with MLease
            </h2>
            <p className="text-gray-600 mb-8">
              MLease empowers users at all enterprise levels to manage, monitor,
              and operationalize ML models effortlessly. With intelligent
              automation, robust monitoring, and accessible tools, MLease
              bridges the gap between advanced AI technologies and practical,
              everyday implementation.
            </p>
            <a
              href="#"
              className="inline-block bg-orange-300 text-gray-700 px-6 py-3 rounded-md hover:bg-orange-400 transition-colors flex items-center"
            >
              Get Started <ChevronRight className="w-4 h-4 ml-1" />
            </a>
          </div>
          <div className="flex justify-center">
            <div className="bg-white p-6 rounded-lg shadow-lg">
              <svg
                viewBox="0 0 200 100"
                fill="none"
                className="w-full h-auto text-orange-500"
              >
                <path
                  d="M50 50C50 30 40 20 20 20C10 20 5 30 5 40C5 50 10 60 20 60C40 60 50 50 50 30"
                  stroke="currentColor"
                  strokeWidth="3"
                />
                <path
                  d="M150 50C150 30 160 20 180 20C190 20 195 30 195 40C195 50 190 60 180 60C160 60 150 50 150 30"
                  stroke="currentColor"
                  strokeWidth="3"
                />
                <circle
                  cx="100"
                  cy="50"
                  r="30"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <text
                  x="100"
                  y="55"
                  textAnchor="middle"
                  fill="currentColor"
                  fontSize="12"
                  fontWeight="bold"
                >
                  MLEASE
                </text>
              </svg>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 px-6 md:px-12 lg:px-24 bg-gray-50">
        <div className="grid md:grid-cols-3 gap-8">
          <div className="bg-blue-500 p-8 rounded-lg text-white text-center">
            <h3 className="text-3xl font-bold mb-4">Experience MLease</h3>
            <p className="mb-4">
              Streamline your ML pipeline with a unified platform designed for
              efficient model deployment.
            </p>
          </div>

          <div className="md:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white p-6 rounded-lg shadow">
              <div className="text-blue-500 mb-4">
                <Code className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-bold mb-2 text-gray-800">
                Seamless Integration
              </h3>
              <p className="text-gray-600">
                Connect and automate your ML workflows effortlessly with robust
                APIs and an intuitive dashboard.
              </p>
            </div>

            <div className="bg-white p-6 rounded-lg shadow">
              <div className="text-blue-500 mb-4">
                <BarChart2 className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-bold mb-2 text-gray-800">
                Real-time Analytics
              </h3>
              <p className="text-gray-600">
                Monitor your model performance live and make informed
                adjustments on the fly.
              </p>
            </div>

            <div className="bg-white p-6 rounded-lg shadow">
              <div className="text-blue-500 mb-4">
                <RefreshCw className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-bold mb-2 text-gray-800">
                Automated Pipelines
              </h3>
              <p className="text-gray-600">
                Accelerate your deployment with end-to-end automation that takes
                your models from training to production seamlessly.
              </p>
            </div>

            <div className="bg-white p-6 rounded-lg shadow">
              <div className="text-blue-500 mb-4">
                <Shield className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-bold mb-2 text-gray-800">
                Enterprise-Grade Security
              </h3>
              <p className="text-gray-600">
                Safeguard your data and models with advanced encryption and
                comprehensive security features.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-12 bg-orange-300 px-6 md:px-12 lg:px-24">
        <div className="text-center">
          <h2 className="text-3xl font-bold text-white mb-6">
            Ready to Simplify Your ML Operations?
          </h2>
          <a
            href="#"
            className="inline-block bg-blue-500 text-white px-8 py-3 rounded-md hover:bg-blue-600 transition-colors"
          >
            Get Started Free
          </a>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-12 px-6 md:px-12 lg:px-24">
        <div className="grid md:grid-cols-4 gap-8">
          <div>
            <h3 className="text-xl font-bold mb-4">MLease</h3>
            <p className="text-gray-400">
              Simplifying machine learning operations for enterprises of all
              sizes.
            </p>
          </div>
          <div>
            <h4 className="font-bold mb-4">Products</h4>
            <ul className="space-y-2 text-gray-400">
              <li>
                <a href="#" className="hover:text-white">
                  MLease Platform
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white">
                  MLease Enterprise
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white">
                  MLease Cloud
                </a>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold mb-4">Resources</h4>
            <ul className="space-y-2 text-gray-400">
              <li>
                <a href="#" className="hover:text-white">
                  Documentation
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white">
                  Blog
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white">
                  Case Studies
                </a>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold mb-4">Company</h4>
            <ul className="space-y-2 text-gray-400">
              <li>
                <a href="#" className="hover:text-white">
                  About Us
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white">
                  Careers
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white">
                  Contact
                </a>
              </li>
            </ul>
          </div>
        </div>
        <div className="border-t border-gray-700 mt-8 pt-8 text-gray-400 text-center">
          <p>&copy; 2025 MLease. All rights reserved.</p>
        </div>
      </footer>

      {/* Scroll to top button */}
      {showScrollButton && (
        <button
          onClick={scrollToTop}
          className="fixed bottom-6 right-6 bg-orange-400 text-white p-3 rounded-full shadow-lg hover:bg-orange-500 transition-colors"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M5 10l7-7m0 0l7 7m-7-7v18"
            />
          </svg>
        </button>
      )}
    </div>
  );
}
