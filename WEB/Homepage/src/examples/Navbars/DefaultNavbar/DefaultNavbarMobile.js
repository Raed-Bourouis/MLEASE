// @mui material components
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Box from "@mui/material/Box";
import { Link } from "react-router-dom";

// Material Kit components
import MKTypography from "components/MKTypography";

// Import your logo
import logo from "assets/images/logo.svg";

function CustomNavbar() {
  return (
    <AppBar position="static" elevation={0} sx={{ backgroundColor: "#FD9D65", color: "white" }}>
      <Toolbar sx={{ justifyContent: "space-between", px: { xs: 2, md: 4 } }}>
        {/* Logo and brand name */}
        <Box display="flex" alignItems="center">
          <IconButton edge="start" color="inherit" component={Link} to="/">
            <img src={logo} alt="MLEase Logo" style={{ height: 30 }} />
          </IconButton>
        </Box>

        {/* Navigation links */}
        <Box display={{ xs: "none", md: "flex" }} alignItems="center">
          <MKTypography
            component={Link}
            to="/products"
            variant="button"
            color="white"
            sx={{ mx: 2, textDecoration: "none", fontWeight: "bold" }}
          >
            Products
          </MKTypography>
          <MKTypography
            component={Link}
            to="/solutions"
            variant="button"
            color="white"
            sx={{ mx: 2, textDecoration: "none", fontWeight: "bold" }}
          >
            Solutions
          </MKTypography>
          <MKTypography
            component={Link}
            to="/resources"
            variant="button"
            color="white"
            sx={{ mx: 2, textDecoration: "none", fontWeight: "bold" }}
          >
            Resources
          </MKTypography>
        </Box>

        {/* Action buttons */}
        <Box display="flex" alignItems="center">
          <Button
            variant="contained"
            color="inherit"
            sx={{
              backgroundColor: "white",
              color: "#FD9D65",
              mr: 1,
              textTransform: "none",
              boxShadow: "none",
              "&:hover": {
                backgroundColor: "#f2f2f2",
              },
            }}
          >
            Login
          </Button>
          <Button
            variant="contained"
            sx={{
              backgroundColor: "#3A8DFF",
              color: "white",
              textTransform: "none",
              boxShadow: "none",
              "&:hover": {
                backgroundColor: "#3278d6",
              },
            }}
          >
            Get Started Free →
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default CustomNavbar;
