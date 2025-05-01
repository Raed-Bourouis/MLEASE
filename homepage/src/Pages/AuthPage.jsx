import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  Eye,
  EyeOff,
  ArrowRight,
  User,
  Lock,
  Mail,
  AlertCircle,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function AuthPage({ isSignIn: propIsSignIn }) {
  const location = useLocation();
  const [isSignIn, setIsSignIn] = useState(
    propIsSignIn ?? location.pathname === "/signin"
  );
  const [showPassword, setShowPassword] = useState(false);
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const navigate = useNavigate();
  const { signin, signup, isLoading, isAuthenticated } = useAuth();

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate("/dashboard");
    }
  }, [isAuthenticated, navigate]);

  // Clear error when form changes
  useEffect(() => {
    setError("");
    setSuccess("");
  }, [isSignIn, email, password, name, username]);

  const toggleForm = () => {
    setIsSignIn(!isSignIn);
    setEmail("");
    setPassword("");
    setName("");
    setUsername("");
    setError("");
    setSuccess("");

    // Update URL without reload
    navigate(isSignIn ? "/signup" : "/signin", { replace: true });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    try {
      if (isSignIn) {
        // Sign In logic
        if (!username || !password) {
          throw new Error("Please enter your username and password");
        }

        const result = await signin(username, password);

        if (result.success) {
          setSuccess("Sign in successful! Redirecting...");
          // Navigate happens automatically via the useEffect
        } else {
          setError(result.error);
        }
      } else {
        // Sign Up logic
        if (!username || !email || !password || !name) {
          throw new Error("Please fill in all fields");
        }

        if (password.length < 6) {
          throw new Error("Password must be at least 6 characters");
        }

        const result = await signup(username, email, password, name);

        if (result.success) {
          setSuccess("Account created successfully! Redirecting...");
          // Navigate happens automatically via the useEffect
        } else {
          setError(result.error);
        }
      }
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10">
          <div className="mb-6">
            <h2 className="text-center text-3xl font-extrabold text-gray-900">
              {isSignIn ? "Sign in to your account" : "Create a new account"}
            </h2>
            <div className="mt-2 text-center text-sm text-gray-600">
              <p>
                {isSignIn
                  ? "Don't have an account?"
                  : "Already have an account?"}
                <button
                  onClick={toggleForm}
                  className="ml-1 font-medium hover:text-orange-400 focus:outline-none focus:underline transition ease-in-out duration-150"
                  style={{ color: "#FD9D65" }}
                >
                  {isSignIn ? "Sign up" : "Sign in"}
                </button>
              </p>
            </div>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-md p-3 flex items-center mb-4">
              <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
              <span className="text-sm text-red-800">{error}</span>
            </div>
          )}

          {success && (
            <div className="bg-green-50 border border-green-200 rounded-md p-3 flex items-center mb-4">
              <AlertCircle className="h-5 w-5 text-green-500 mr-2" />
              <span className="text-sm text-green-800">{success}</span>
            </div>
          )}

          <div className="space-y-6">
            {!isSignIn && (
              <div>
                <label
                  htmlFor="name"
                  className="block text-sm font-medium text-gray-700"
                >
                  Full Name
                </label>
                <div className="mt-1 relative rounded-md shadow-sm">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <User className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    id="name"
                    name="name"
                    type="text"
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 sm:text-sm"
                    placeholder="John Doe"
                    style={{
                      "--tw-ring-color": "#FD9D65",
                      "--tw-border-opacity": 1,
                    }}
                    disabled={isLoading}
                  />
                </div>
              </div>
            )}

            <div>
              <label
                htmlFor="username"
                className="block text-sm font-medium text-gray-700"
              >
                Username
              </label>
              <div className="mt-1 relative rounded-md shadow-sm">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <User className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="username"
                  name="username"
                  type="text"
                  autoComplete="username"
                  required
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 sm:text-sm"
                  placeholder="johndoe"
                  style={{
                    "--tw-ring-color": "#FD9D65",
                    "--tw-border-opacity": 1,
                  }}
                  disabled={isLoading}
                />
              </div>
            </div>

            {!isSignIn && (
              <div>
                <label
                  htmlFor="email"
                  className="block text-sm font-medium text-gray-700"
                >
                  Email address
                </label>
                <div className="mt-1 relative rounded-md shadow-sm">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Mail className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    autoComplete="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 sm:text-sm"
                    placeholder="you@example.com"
                    style={{
                      "--tw-ring-color": "#FD9D65",
                      "--tw-border-opacity": 1,
                    }}
                    disabled={isLoading}
                  />
                </div>
              </div>
            )}

            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium text-gray-700"
              >
                Password
              </label>
              <div className="mt-1 relative rounded-md shadow-sm">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="password"
                  name="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete={isSignIn ? "current-password" : "new-password"}
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="block w-full pl-10 pr-10 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 sm:text-sm"
                  placeholder="••••••••"
                  style={{
                    "--tw-ring-color": "#FD9D65",
                    "--tw-border-opacity": 1,
                  }}
                  disabled={isLoading}
                />
                <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="text-gray-400 hover:text-gray-500 focus:outline-none"
                  >
                    {showPassword ? (
                      <EyeOff className="h-5 w-5" />
                    ) : (
                      <Eye className="h-5 w-5" />
                    )}
                  </button>
                </div>
              </div>
            </div>

            {isSignIn && (
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <input
                    id="remember-me"
                    name="remember-me"
                    type="checkbox"
                    className="h-4 w-4 rounded border-gray-300 focus:ring-2"
                    style={{
                      "--tw-ring-color": "#FD9D65",
                      "--tw-border-opacity": 1,
                    }}
                  />
                  <label
                    htmlFor="remember-me"
                    className="ml-2 block text-sm text-gray-700"
                  >
                    Remember me
                  </label>
                </div>

                <div className="text-sm">
                  <a
                    href="#"
                    className="font-medium hover:text-orange-400"
                    style={{ color: "#FD9D65" }}
                  >
                    Forgot your password?
                  </a>
                </div>
              </div>
            )}

            <div>
              <button
                onClick={handleSubmit}
                className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white hover:bg-orange-600 focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-200"
                style={{
                  backgroundColor: "#FD9D65",
                  "--tw-ring-color": "#FD9D65",
                }}
                disabled={isLoading}
              >
                {isLoading ? (
                  <svg
                    className="animate-spin h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                ) : (
                  <>
                    <span className="absolute left-0 inset-y-0 flex items-center pl-3">
                      <ArrowRight
                        className="h-5 w-5 group-hover:text-orange-300"
                        style={{ color: "#FFF" }}
                      />
                    </span>
                    {isSignIn ? "Sign in" : "Sign up"}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
