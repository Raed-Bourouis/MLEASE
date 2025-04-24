/** @jsxImportSource @emotion/react */
import React from "react";
import PropTypes from "prop-types";
import { Container, Grid, Box } from "@mui/material";
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";
import MKButton from "components/MKButton";
import { keyframes } from "@emotion/react";
import {
  FaArrowRight,
  FaPython,
  FaDocker,
  FaRocket,
  FaDatabase,
  FaStickyNote,
} from "react-icons/fa";

function IconCircle({ icon }) {
  return (
    <Box
      sx={{
        backgroundColor: "#fff",
        color: "#F6B035",
        width: 70,
        height: 70,
        borderRadius: "50%",
        boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "3rem",
      }}
    >
      {icon}
    </Box>
  );
}

IconCircle.propTypes = {
  icon: PropTypes.node.isRequired,
};

// Animations
const gradientAnimation = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

const textFadeAnimation = keyframes`
  0%, 100% { opacity: 1; }
  50% { opacity: 0.85; }
`;

export default function Download() {
  return (
    <MKBox component="section">
      {/* HERO SECTION */}
      <MKBox
        sx={{
          position: "relative",
          minHeight: { xs: "50vh", md: "60vh" },
          background:
            "linear-gradient(270deg,rgb(245, 144, 107), #FF8C00,rgb(223, 114, 24), #FF6347)",

          backgroundSize: "600% 600%",
          animation: `${gradientAnimation} 30s ease infinite`,
          overflow: "hidden",
        }}
      >
        {/* noise texture overlay for depth */}
        <Box
          sx={{
            position: "absolute",
            width: "100%",
            height: "100%",
            top: 0,
            left: 0,
            backgroundImage: `url("https://www.transparenttextures.com/patterns/asfalt-light.png")`,
            opacity: 0.05,
            zIndex: 1,
          }}
        />

        {/* Content */}
        <Container
          sx={{
            position: "relative",
            zIndex: 2,
            pt: { xs: 8, md: 12 },
            pb: { xs: 4, md: 8 },
            textAlign: "center",
            color: "#fff",
          }}
        >
          <MKTypography
            variant="h2"
            fontWeight="bold"
            sx={{
              mb: 2,
              fontSize: { xs: "2.5rem", md: "3.5rem" },
              color: "#FFFFFF",
              textShadow: "2px 2px 4px rgba(0,0,0,0.5)",
              animation: `${textFadeAnimation} 6s ease-in-out infinite`,
              fontFamily: "Arial, sans-serif",
            }}
          >
            Your work, everywhere{" "}
            <MKBox component="span" sx={{ position: "relative", display: "inline-block" }}>
              <MKBox
                component="span"
                sx={{
                  position: "absolute",
                  left: 0,
                  bottom: 4,
                  width: "100%",
                  height: "30px",
                  bgcolor: "#3A8DFF",
                  zIndex: -1,
                  borderRadius: 1,
                  filter: "drop-shadow(0 0 8px #3A8DFF)",
                }}
              />
              you are
            </MKBox>
          </MKTypography>

          <MKTypography
            variant="body1"
            sx={{
              mb: 4,
              fontSize: { xs: "1.1rem", md: "1.3rem" },
              maxWidth: 650,
              mx: "auto",
              lineHeight: 1.6,
              color: "#FFFFFF",
              textShadow: "1px 1px 3px rgba(0,0,0,0.4)", // Subtle text shadow for readability
              animation: `${textFadeAnimation} 8s ease-in-out infinite`, // Optional text fade
              fontFamily: "Arial, sans-serif",
            }}
          >
            An MLOps-driven platform that simplifies and democratizes machine learning model
            deployment. Seamlessly manage, monitor, and operationalize your ML models with
            cutting-edge automation and an intuitive workflow.
          </MKTypography>

          <MKButton
            variant="contained"
            size="large"
            sx={{
              backgroundColor: "#3A8DFF",
              color: "#fff",
              textTransform: "none",
              boxShadow: "none",
              fontSize: { xs: "1rem", md: "1.25rem" },
              px: { xs: 3, md: 4 },
              py: { xs: 1, md: 1.5 },
              "&:hover": { backgroundColor: "#3278d6" },
            }}
            endIcon={<FaArrowRight />}
          >
            Get Started →
          </MKButton>
        </Container>
      </MKBox>

      {/* TECHNOLOGIES & DATA SECTION */}
      <MKBox sx={{ backgroundColor: "#fff", py: { xs: 6, md: 10 } }}>
        <Container>
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={6}>
              <MKTypography
                variant="h4"
                fontWeight="bold"
                sx={{
                  mb: 2,
                  fontSize: { xs: "1.8rem", md: "2.2rem" },
                  color: "#000",
                }}
              >
                100% Your Data
              </MKTypography>

              <MKTypography
                variant="body1"
                sx={{
                  mb: 4,
                  fontSize: { xs: "1.1rem", md: "1.3rem" },
                  color: "#555",
                  lineHeight: 1.6,
                  maxWidth: 550,
                }}
              >
                Our platform is fully open source—ensuring your ML models, data, and insights remain
                entirely accessible and secure with state-of-the-art end-to-end encryption.
              </MKTypography>

              <MKButton
                variant="contained"
                size="large"
                sx={{
                  backgroundColor: "#3A8DFF",
                  color: "#fff",
                  textTransform: "none",
                  boxShadow: "none",
                  fontSize: { xs: "1rem", md: "1.25rem" },
                  px: { xs: 3, md: 4 },
                  py: { xs: 1, md: 1.5 },
                  "&:hover": { backgroundColor: "#3278d6" },
                }}
                endIcon={<FaArrowRight />}
              >
                Learn More →
              </MKButton>
            </Grid>

            <Grid item xs={12} md={6}>
              <Box
                sx={{
                  position: "relative",
                  width: "100%",
                  height: { xs: 300, md: 400 },
                  mx: "auto",
                }}
              >
                <Box
                  component="svg"
                  viewBox="0 0 400 400"
                  sx={{
                    width: "100%",
                    height: "100%",
                    position: "absolute",
                    top: 0,
                    left: 0,
                  }}
                >
                  <line
                    x1="200"
                    y1="200"
                    x2="100"
                    y2="70"
                    stroke="#FF7F50"
                    strokeWidth="3"
                    strokeDasharray="6,4"
                  />
                  <line
                    x1="200"
                    y1="200"
                    x2="300"
                    y2="110"
                    stroke="#FF7F50"
                    strokeWidth="3"
                    strokeDasharray="6,4"
                  />
                  <line
                    x1="200"
                    y1="200"
                    x2="70"
                    y2="270"
                    stroke="#FF7F50"
                    strokeWidth="3"
                    strokeDasharray="6,4"
                  />
                  <line
                    x1="200"
                    y1="200"
                    x2="330"
                    y2="270"
                    stroke="#FF7F50"
                    strokeWidth="3"
                    strokeDasharray="6,4"
                  />
                </Box>

                {/* Technology Icons */}
                <Box
                  sx={{
                    position: "absolute",
                    top: "20%",
                    left: "30%",
                    transform: "translate(-50%, -50%)",
                  }}
                >
                  <IconCircle icon={<FaPython />} />
                </Box>
                <Box
                  sx={{
                    position: "absolute",
                    top: "30%",
                    right: "20%",
                    transform: "translate(-40%, -70%)",
                  }}
                >
                  <IconCircle icon={<FaDocker />} />
                </Box>
                <Box
                  sx={{
                    position: "absolute",
                    bottom: "25%",
                    left: "25%",
                    transform: "translate(-50%, 50%)",
                  }}
                >
                  <IconCircle icon={<FaDatabase />} />
                </Box>
                <Box
                  sx={{
                    position: "absolute",
                    bottom: "20%",
                    right: "30%",
                    transform: "translate(50%, 50%)",
                  }}
                >
                  <IconCircle icon={<FaRocket />} />
                </Box>
                <Box
                  sx={{
                    position: "absolute",
                    top: "50%",
                    left: "50%",
                    transform: "translate(-50%, -50%)",
                  }}
                >
                  <IconCircle icon={<FaStickyNote />} />
                </Box>
              </Box>
            </Grid>
          </Grid>
        </Container>
      </MKBox>
    </MKBox>
  );
}
