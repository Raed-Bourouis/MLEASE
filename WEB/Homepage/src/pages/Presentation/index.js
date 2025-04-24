// Presentation.jsx

// Imports
import { useState, useEffect } from "react";
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import IconButton from "@mui/material/IconButton";
import Button from "@mui/material/Button";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import Brightness4Icon from "@mui/icons-material/Brightness4";
import Brightness7Icon from "@mui/icons-material/Brightness7";

import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";
import DefaultNavbar from "examples/Navbars/DefaultNavbar";
import DefaultFooter from "examples/Footers/DefaultFooter";
import routes from "routes";
import footerRoutes from "footer.routes";

import Counters from "pages/Presentation/sections/Counters";
import Information from "pages/Presentation/sections/Information";
import Pages from "pages/Presentation/sections/Pages";
import Download from "pages/Presentation/sections/Download";
import BuiltByDevelopers from "pages/Presentation/components/BuiltByDevelopers";
import FilledInfoCard from "examples/Cards/InfoCards/FilledInfoCard";

import { motion } from "framer-motion";

import aiImage from "assets/images/background_ai.png"; // ✅ Your AI image

function Presentation() {
  const [mode, setMode] = useState("light");
  const isDarkMode = mode === "dark";

  const [showButton, setShowButton] = useState(false);
  useEffect(() => {
    const handleScroll = () => {
      setShowButton(window.scrollY > 300);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <MKBox sx={{ fontFamily: "Roboto, sans-serif", backgroundColor: "#fff" }}>
      {/* Navbar */}
      <DefaultNavbar
        routes={routes}
        action={{
          type: "external",
          route: "https://www.creative-tim.com/product/material-kit-react",
          label: "Free Download",
          color: "info",
        }}
        sticky
      />

      {/* Hero Section */}
      <MKBox
        component="section"
        minHeight="100vh"
        width="100%"
        sx={{
          position: "relative",
          display: "flex",
          alignItems: "center",
          background: "linear-gradient(135deg, #FD9D65, #FFB88C)",
          overflow: "hidden",
          px: { xs: 4, md: 12 },
          py: { xs: 10, md: 20 },
        }}
      >
        {/* Abstract Signal Background SVG */}
        <MKBox
          component="svg"
          viewBox="0 0 1440 320"
          sx={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            opacity: 0.2,
            zIndex: 1,
          }}
        >
          <path
            fill="#ffffff"
            fillOpacity="0.3"
            d="M0,160L60,154.7C120,149,240,139,360,160C480,181,600,235,720,250.7C840,267,960,245,1080,229.3C1200,213,1320,203,1380,197.3L1440,192L1440,320L1380,320C1320,320,1200,320,1080,320C960,320,840,320,720,320C600,320,480,320,360,320C240,320,120,320,60,320L0,320Z"
          ></path>
        </MKBox>

        {/* Theme Toggle */}
        <IconButton
          sx={{ position: "absolute", top: 16, right: 16, zIndex: 10 }}
          onClick={() => setMode((prev) => (prev === "light" ? "dark" : "light"))}
          color="inherit"
        >
          {isDarkMode ? <Brightness7Icon /> : <Brightness4Icon />}
        </IconButton>

        <Container sx={{ position: "relative", zIndex: 2 }}>
          <Grid container spacing={6} alignItems="center">
            {/* Left Side: Text */}
            <Grid item xs={12} md={6}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
              >
                <MKTypography
                  variant="h1"
                  sx={{
                    color: "#fff",
                    fontWeight: "bold",
                    mb: 4,
                    lineHeight: 1.2,
                    fontSize: { xs: "2.5rem", md: "3.8rem" },
                    textShadow: "2px 2px 6px rgba(0,0,0,0.4)",
                  }}
                >
                  Getting Started with <br />
                  MLEASE !
                </MKTypography>

                <MKTypography
                  variant="body1"
                  sx={{
                    color: "#fff",
                    opacity: 0.95,
                    lineHeight: 1.8,
                    mb: 5,
                    fontSize: { xs: "1.2rem", md: "1.4rem" },
                  }}
                >
                  MLEASE is an MLOps-driven platform that simplifies and democratizes machine
                  learning model deployment.
                </MKTypography>

                <Button
                  variant="contained"
                  sx={{
                    backgroundColor: "#4D7CFE",
                    color: "#fff",
                    textTransform: "none",
                    fontWeight: "bold",
                    borderRadius: "8px",
                    padding: "14px 28px",
                    fontSize: "1.1rem",
                    transition: "all 0.3s ease",
                    "&:hover": {
                      backgroundColor: "#3a6de0",
                      transform: "translateY(-2px)",
                      boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
                    },
                  }}
                >
                  Get Started Free →
                </Button>
              </motion.div>
            </Grid>

            {/* Right Side: Image */}
            <Grid item xs={12} md={6} display="flex" justifyContent="center">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 1 }}
              >
                <MKBox
                  component="img"
                  src={aiImage}
                  alt="AI Visual"
                  sx={{
                    width: { xs: "95%", md: "85%" }, // ✅ Bigger!
                    borderRadius: "16px",
                    boxShadow: "0 8px 30px rgba(0,0,0,0.3)",
                    background: "#fff",
                    padding: "16px",
                    transition: "transform 0.3s ease",
                    "&:hover": {
                      transform: "scale(1.05)",
                    },
                  }}
                />
              </motion.div>
            </Grid>
          </Grid>
        </Container>
      </MKBox>

      {/* Divider */}
      <MKBox sx={{ height: "6px", backgroundColor: "#FD9D65", opacity: 0.4 }} />

      {/* Main Sections */}
      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 1 }}
      >
        <MKBox py={8}>
          <Counters />
        </MKBox>

        <MKBox py={8} bgcolor="#fff">
          <Information />
        </MKBox>

        <MKBox py={8}>
          <Pages />
        </MKBox>

        <MKBox py={8} bgcolor="#fff">
          <Container>
            <BuiltByDevelopers />
          </Container>

          <Container>
            <Grid container spacing={4}>
              {[
                {
                  icon: "flag",
                  title: "Getting Started",
                  description: "Check how to work with MLEASE.",
                  link: "https://www.creative-tim.com/learning-lab/react/overview/material-kit/",
                },
                {
                  icon: "precision_manufacturing",
                  title: "Plugins",
                  description: "Discover plugins used to build the platform.",
                  link: "https://www.creative-tim.com/learning-lab/react/overview/datepicker/",
                },
                {
                  icon: "apps",
                  title: "Components",
                  description: "Use pre-made components to build faster.",
                  link: "https://www.creative-tim.com/learning-lab/react/alerts/material-kit/",
                },
              ].map((item, index) => (
                <Grid item xs={12} lg={4} key={index}>
                  <FilledInfoCard
                    color="info"
                    icon={item.icon}
                    title={item.title}
                    description={item.description}
                    action={{
                      type: "external",
                      route: item.link,
                      label: "Read more",
                    }}
                  />
                </Grid>
              ))}
            </Grid>
          </Container>

          <MKBox pt={8}>
            <Download />
          </MKBox>
        </MKBox>
      </motion.div>

      {/* Footer */}
      <MKBox pt={8} px={1} mt={8}>
        <DefaultFooter content={footerRoutes} />
      </MKBox>

      {/* Scroll to Top */}
      {showButton && (
        <IconButton
          onClick={scrollToTop}
          sx={{
            position: "fixed",
            bottom: 24,
            right: 24,
            backgroundColor: "#FD9D65",
            color: "#fff",
            boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
            "&:hover": {
              backgroundColor: "#e07b3a",
            },
            zIndex: 1000,
          }}
        >
          <KeyboardArrowUpIcon />
        </IconButton>
      )}
    </MKBox>
  );
}

export default Presentation;
