// @mui material components
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import Button from "@mui/material/Button";

// Material Kit 2 React components
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";
import logo from "assets/images/logo.svg";

function MLOpsHeroSection() {
  return (
    <MKBox
      component="section"
      py={{ xs: 8, md: 12 }}
      sx={{
        backgroundColor: "#fff",
        backgroundSize: "auto",
      }}
    >
      <Container>
        <Grid container spacing={4} alignItems="center">
          {/* Left side: Text */}
          <Grid item xs={12} md={6}>
            <MKTypography
              variant="h3"
              sx={{
                color: "#212121",
                fontWeight: "bold",
                mb: 2,
                fontFamily: "Arial, sans-serif",
              }}
            >
              Simplify AI Operations with MLease
            </MKTypography>

            <MKTypography
              variant="body1"
              sx={{
                color: "#555",
                mb: 3,
                lineHeight: 1.7,
                fontSize: "1.1rem",
                fontFamily: "Arial, sans-serif",
              }}
            >
              <strong>MLEase</strong> empowers users at all enterprise levels to manage, monitor,
              and operationalize ML models effortlessly. With intelligent automation, robust
              monitoring, and accessible tools, MLease bridges the gap between advanced AI
              technologies and practical, everyday implementation.
            </MKTypography>

            <Button
              variant="contained"
              sx={{
                backgroundColor: "#FD9D65",
                color: "#fff",
                textTransform: "none",
                fontSize: "1rem",
                padding: "10px 20px",
                boxShadow: "0 4px 10px rgba(253, 157, 101, 0.5)",
                "&:hover": {
                  backgroundColor: "#e68a54",
                },
              }}
            >
              Get Started →
            </Button>
          </Grid>

          {/* Right side: Visual */}
          <Grid item xs={12} md={6} display="flex" justifyContent="center">
            <MKBox
              component="img"
              src={logo}
              alt="AI powered by MLease"
              sx={{
                width: "100%",
                maxWidth: "400px",
                borderRadius: "16px",
                boxShadow: "0 10px 20px rgba(0,0,0,0.15)",
                objectFit: "cover",
              }}
            />
          </Grid>
        </Grid>
      </Container>
    </MKBox>
  );
}

export default MLOpsHeroSection;
