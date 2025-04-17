// CustomFooter.jsx

import React from "react";
import { Container, Grid, Box } from "@mui/material";
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";

export default function CustomFooter() {
  return (
    <MKBox
      component="footer"
      sx={{
        mt: 10,
        pt: 8,
        pb: 4,
        backgroundColor: "#FD9D65",
        fontFamily: "Roboto, sans-serif",
        color: "white",
      }}
    >
      {/* Our Sponsors Section */}
      <Container sx={{ mb: 8 }}>
        <MKTypography
          variant="h4"
          textAlign="center"
          fontWeight="bold"
          sx={{ mb: 4, fontFamily: "Roboto, sans-serif" }}
        >
          Our{" "}
          <Box component="span" sx={{ background: "#FFE492", px: 1, borderRadius: 1 }}>
            sponsors
          </Box>
        </MKTypography>
        <Grid container justifyContent="center" spacing={4}>
          <Grid item>
            <Box
              component="img"
              src="https://cdn-icons-png.flaticon.com/512/732/732221.png"
              alt="Apple"
              sx={{ height: 40 }}
            />
          </Grid>
          <Grid item>
            <Box
              component="img"
              src="https://cdn-icons-png.flaticon.com/512/732/732228.png"
              alt="Microsoft"
              sx={{ height: 40 }}
            />
          </Grid>
          <Grid item>
            <Box
              component="img"
              src="https://cdn-icons-png.flaticon.com/512/2111/2111615.png"
              alt="Slack"
              sx={{ height: 40 }}
            />
          </Grid>
          <Grid item>
            <Box
              component="img"
              src="https://cdn-icons-png.flaticon.com/512/300/300221.png"
              alt="Google"
              sx={{ height: 40 }}
            />
          </Grid>
        </Grid>
      </Container>

      {/* Footer Main Content */}
      <Container>
        <Grid container spacing={4} justifyContent="center" textAlign="center">
          <Grid item xs={12} md={6}>
            <MKTypography
              variant="h4"
              fontWeight="bold"
              sx={{ fontFamily: "Roboto, sans-serif", mb: 1 }}
            >
              Try MLEASE <br /> today
            </MKTypography>
            <MKTypography variant="body2" sx={{ mb: 3, fontSize: "1rem", opacity: 0.9 }}>
              Get started for free. MLEASE is by your side as your needs grow.
            </MKTypography>
            <Box>
              <Box
                component="a"
                href="#"
                sx={{
                  px: 3,
                  py: 1,
                  fontSize: "0.9rem",
                  backgroundColor: "#3A8DFF",
                  color: "white",
                  borderRadius: 2,
                  textDecoration: "none",
                }}
              >
                Try for free →
              </Box>
            </Box>
            <MKTypography variant="body2" sx={{ mt: 3 }}>
              Contact Us
            </MKTypography>
            <Box sx={{ mt: 1, display: "flex", gap: 2, justifyContent: "center" }}>
              <Box component="a" href="#" color="white">
                <i className="fab fa-facebook-f" style={{ fontSize: 24 }}></i>
              </Box>
              <Box component="a" href="#" color="white">
                <i className="fab fa-twitter" style={{ fontSize: 24 }}></i>
              </Box>
              <Box component="a" href="#" color="white">
                <i className="fab fa-linkedin-in" style={{ fontSize: 24 }}></i>
              </Box>
            </Box>
          </Grid>
        </Grid>

        {/* Bottom Footer Grid */}
        <Grid container spacing={4} justifyContent="space-between" mt={8}>
          <Grid item xs={12} md={4}>
            <MKTypography
              variant="body1"
              fontWeight="bold"
              sx={{ fontFamily: "Roboto, sans-serif", mb: 1 }}
            >
              MLEASE
            </MKTypography>
            <MKTypography variant="caption">
              whitespace was created for the new dependencies on ML. We make a better workspace
              around the world.
            </MKTypography>
          </Grid>
          <Grid item xs={12} md={4}>
            <MKTypography
              variant="body1"
              fontWeight="bold"
              sx={{ fontFamily: "Roboto, sans-serif", mb: 1 }}
            >
              Company
            </MKTypography>
            <MKTypography variant="caption" display="block">
              About us
            </MKTypography>
            <MKTypography variant="caption" display="block">
              Media kit
            </MKTypography>
          </Grid>
          <Grid item xs={12} md={4} textAlign="right">
            <MKTypography
              variant="body1"
              fontWeight="bold"
              sx={{ fontFamily: "Roboto, sans-serif", mb: 1 }}
            >
              Try It Today
            </MKTypography>
            <MKTypography variant="caption" display="block">
              Get started for free.
            </MKTypography>
            <Box
              component="a"
              href="#"
              sx={{
                mt: 1,
                display: "inline-block",
                px: 3,
                py: 1,
                fontSize: "0.9rem",
                backgroundColor: "#3A8DFF",
                color: "white",
                borderRadius: 2,
                textDecoration: "none",
              }}
            >
              Start today →
            </Box>
          </Grid>
        </Grid>
      </Container>
    </MKBox>
  );
}
