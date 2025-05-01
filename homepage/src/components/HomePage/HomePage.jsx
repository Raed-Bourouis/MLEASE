import { Code, RefreshCw, Shield, BarChart3, Pointer } from "lucide-react";
import React, { useState } from "react";
import Navbar from "./Navbar";

import AIphoto from "../../assets/background_ai.png";
import mlease_logo from "../../assets/logo.svg";
import RotatingFront from "../../assets/rotating-card-bg-front.jpeg";
import RotatingBack from "../../assets/rotating-card-bg-back.jpeg";
import { motion } from "framer-motion";

const HomePage = () => {
  return (
    <>
      <Navbar />
      <MLeaseLanding />
      <HeroSection />
      <FeaturesSection />
      <PageShowcase />
    </>
  );
};

export default HomePage;

function MLeaseLanding() {
  return (
    <div className="relative  bg-gradient-to-br from-[#FD9D65] to-[#FFB88C] w-full h-main  flex items-center justify-center p-6 overflow-hidden">
      <svg
        className="absolute top-0 left-0 w-full h-full z-0 opacity-35"
        viewBox="0 0 1440 320"
        preserveAspectRatio="none"
      >
        <path
          fill="#ffffff"
          fillOpacity="0.3"
          d="M0,160L60,154.7C120,149,240,139,360,160C480,181,600,235,720,250.7C840,267,960,245,1080,229.3C1200,213,1320,203,1380,197.3L1440,192L1440,320L1380,320C1320,320,1200,320,1080,320C960,320,840,320,720,320C600,320,480,320,360,320C240,320,120,320,60,320L0,320Z"
        ></path>
      </svg>

      <div className="relative z-10 max-w-6xl w-full flex flex-col md:flex-row items-center justify-between gap-8">
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

function HeroSection() {
  return (
    <div className="flex flex-col lg:flex-row items-center justify-around w-full h-main bg-white py-16 px-4 sm:px-6 lg:px-8">
      {/* Left side: Text content */}
      <div className="max-w-xl mb-10 lg:mb-0">
        <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
          Simplify AI Operations with MLease
        </h1>
        <div className="mb-8">
          <span className="font-semibold text-gray-700">MLEASE </span>
          <p className="text-gray-600 mt-2 inline">
            empowers users at all enterprise levels to manage, monitor, and
            operationalize ML models effortlessly. With intelligent automation,
            robust monitoring, and accessible tools, MLease bridges the gap
            between advanced AI technologies and practical, everyday
            implementation.
          </p>
        </div>
        <button className="bg-orange-300 hover:bg-orange-400 text-white py-2 px-6 rounded-md cursor-pointer transition-colors">
          Get Started →
        </button>
      </div>

      {/* Right side: Logo */}
      <div className="bg-white p-8 rounded-lg shadow-lg">
        <img src={mlease_logo} alt="MLease Logo" className="w-64 h-auto" />
      </div>
    </div>
  );
}

function FeaturesSection() {
  // State to track which card is flipped
  const [flippedCard, setFlippedCard] = useState(false);

  // Features data
  const features = [
    {
      id: "integration",
      icon: <Code className="w-6 h-6 text-blue-500" />,
      title: "Seamless Integration",
      description:
        "Connect and automate your ML workflows effortlessly with robust APIs and an intuitive dashboard.",
    },
    {
      id: "analytics",
      icon: <BarChart3 className="w-6 h-6 text-blue-500" />,
      title: "Real-time Analytics",
      description:
        "Monitor your model performance live and make informed adjustments on the fly.",
    },
    {
      id: "pipelines",
      icon: <RefreshCw className="w-6 h-6 text-blue-500" />,
      title: "Automated Pipelines",
      description:
        "Accelerate your deployment with end-to-end automation that takes your models from training to production seamlessly.",
    },
    {
      id: "security",
      icon: <Shield className="w-6 h-6 text-blue-500" />,
      title: "Enterprise-Grade Security",
      description:
        "Safeguard your data and models with advanced encryption and comprehensive security features.",
    },
  ];

  return (
    <div className="w-full bg-white py-16 px-4 h-main sm:px-6 lg:px-8 flex flex-col lg:flex-row items-center gap-x-24 justify-center">
      {/* Left side: Rotating cards */}
      <div className=" h-[65dvh] w-[50dvh] flex justify-center items-center p-0  rounded-2xl">
        <div className="group [perspective:1000px] h-full w-full  rounded-2xl">
          <div className="relative  rounded-2xl w-full h-full transition-transform duration-700 [transform-style:preserve-3d] group-hover:[transform:rotateY(180deg)]">
            {/* Front Side */}
            <div
              style={{ backgroundImage: `url(${RotatingFront})` }}
              className="absolute inset-0 text-white rounded-2xl shadow-xl flex flex-col justify-center bg-cover bg-center items-center [backface-visibility:hidden]"
            >
              <div className="absolute inset-0 bg-[linear-gradient(195deg,rgba(73,163,241,0.85),rgba(73,163,241,0.85))]  rounded-2xl  z-0" />
              <div className="w-full h-full flex justify-center relative  items-center flex-col">
                <Pointer className="mb-16" size={50} />
                <h2 className="text-2xl font-bold">Experience MLease</h2>
                <p className="mt-2 text-center px-6">
                  Streamline your ML pipeline with a unified platform designed
                  for efficient model deployment.
                </p>
              </div>
            </div>

            {/* Back Side */}
            <div
              style={{ backgroundImage: `url(${RotatingBack})` }}
              className="absolute inset-0 text-white rounded-2xl shadow-xl flex flex-col justify-center items-center [transform:rotateY(180deg)] bg-cover bg-center [backface-visibility:hidden]"
            >
              <div className="absolute inset-0  bg-[linear-gradient(195deg,rgba(73,163,241,0.85),rgba(73,163,241,0.85))] rounded-2xl z-0" />
              <div className="w-full h-full flex justify-center relative items-center flex-col">
                <h2 className="text-2xl font-bold">Discover MLease</h2>
                <p className="mt-2 text-center px-6">
                  Dive into our intuitive dashboard, access real-time analytics,
                  and unlock end-to-end automation for your ML models.
                </p>
                <button className="mt-6 px-4 py-2 bg-white text-blue-600 font-semibold rounded-xl shadow hover:bg-gray-100 transition">
                  Explore Features
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* Right side: Features grid */}
      <div className="w-1/2 lg:w-1/2 grid md:grid-cols-2 gap-x-8 gap-y-12">
        {features.map((feature) => (
          <div key={feature.id} className="flex flex-col  text-lg">
            <div className="mb-4 bg-blue-100 w-10 h-10 rounded  flex items-center justify-center">
              {feature.icon}
            </div>
            <h3 className="text-2xl font-bold  text-gray-800 mb-2">
              {feature.title}
            </h3>
            <p className="text-gray-600">{feature.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function PageShowcase({
  // You can import and pass these images as props
  aboutUsImage,
  contactUsImage,
  signInImage,
  authorImage,
}) {
  return (
    <div className="w-full bg-white py-16 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col lg:flex-row items-start justify-between gap-12">
          {/* Left side: Grid of page previews */}
          <div className="w-full lg:w-3/5 grid md:grid-cols-2 gap-6">
            {/* About Us Page */}
            <div className="flex flex-col">
              <div className="shadow-lg rounded-lg overflow-hidden mb-3 hover:shadow-xl transition-shadow">
                {aboutUsImage ? (
                  <img
                    src={aboutUsImage}
                    alt="About Us Page Preview"
                    className="w-full"
                  />
                ) : (
                  <div className="bg-gray-200 w-full h-64 flex items-center justify-center">
                    <span className="text-gray-500">About Us Image</span>
                  </div>
                )}
              </div>
              <h3 className="text-lg font-medium text-gray-800">
                About Us Page
              </h3>
            </div>

            {/* Contact Us Page */}
            <div className="flex flex-col">
              <div className="shadow-lg rounded-lg overflow-hidden mb-3 hover:shadow-xl transition-shadow">
                {contactUsImage ? (
                  <img
                    src={contactUsImage}
                    alt="Contact Us Page Preview"
                    className="w-full"
                  />
                ) : (
                  <div className="bg-gray-200 w-full h-64 flex items-center justify-center">
                    <span className="text-gray-500">Contact Us Image</span>
                  </div>
                )}
              </div>
              <h3 className="text-lg font-medium text-gray-800">
                Contact Us Page
              </h3>
            </div>

            {/* Sign In Page */}
            <div className="flex flex-col">
              <div className="shadow-lg rounded-lg overflow-hidden mb-3 hover:shadow-xl transition-shadow">
                {signInImage ? (
                  <img
                    src={signInImage}
                    alt="Sign In Page Preview"
                    className="w-full"
                  />
                ) : (
                  <div className="bg-gray-200 w-full h-64 flex items-center justify-center">
                    <span className="text-gray-500">Sign In Image</span>
                  </div>
                )}
              </div>
              <h3 className="text-lg font-medium text-gray-800">
                Sign In Page
              </h3>
            </div>

            {/* Author Page */}
            <div className="flex flex-col">
              <div className="shadow-lg rounded-lg overflow-hidden mb-3 hover:shadow-xl transition-shadow">
                {authorImage ? (
                  <img
                    src={authorImage}
                    alt="Author Page Preview"
                    className="w-full"
                  />
                ) : (
                  <div className="bg-gray-200 w-full h-64 flex items-center justify-center">
                    <span className="text-gray-500">Author Image</span>
                  </div>
                )}
              </div>
              <h3 className="text-lg font-medium text-gray-800">Author Page</h3>
            </div>
          </div>

          {/* Right side: Description */}
          <div className="w-full lg:w-2/5 h-[70dvh] flex flex-col justify-center">
            <h2 className="text-4xl font-bold text-gray-800 mb-6">
              Empowering MLOps, Simplified
            </h2>
            <p className="text-gray-600 text-lg">
              MLease empowers users of all expertise levels to manage, monitor,
              and operationalize ML models effectively with automation,
              end-to-end support, and accessible tools—bridging the gap between
              advanced ML technologies and practical implementation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

const FlipCard = () => {
  return (
    <div className="w-full flex justify-center items-center p-4">
      <div className="group [perspective:1000px]">
        <div className="relative w-80 h-96 transition-transform duration-700 [transform-style:preserve-3d] group-hover:[transform:rotateY(180deg)]">
          {/* Front Side */}
          <div className="absolute inset-0 bg-blue-500 text-white rounded-2xl shadow-xl flex flex-col justify-center items-center [backface-visibility:hidden]">
            <div className="text-5xl mb-4">🖱️</div>
            <h2 className="text-2xl font-bold">Experience MLease</h2>
            <p className="mt-2 text-center px-6">
              Streamline your ML pipeline with a unified platform designed for
              efficient model deployment.
            </p>
          </div>

          {/* Back Side */}
          <div className="absolute inset-0 bg-blue-600 text-white rounded-2xl shadow-xl flex flex-col justify-center items-center [transform:rotateY(180deg)] [backface-visibility:hidden]">
            <h2 className="text-2xl font-bold">Discover MLease</h2>
            <p className="mt-2 text-center px-6">
              Dive into our intuitive dashboard, access real-time analytics, and
              unlock end-to-end automation for your ML models.
            </p>
            <button className="mt-6 px-4 py-2 bg-white text-blue-600 font-semibold rounded-xl shadow hover:bg-gray-100 transition">
              Explore Features
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
