import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import PropTypes from "prop-types";

// @mui material components
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Container from "@mui/material/Container";
import Box from "@mui/material/Box";
import Icon from "@mui/material/Icon";
import Drawer from "@mui/material/Drawer";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import Divider from "@mui/material/Divider";
import Button from "@mui/material/Button";

// Material Kit 2 React components
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";
import MKButton from "components/MKButton";

// Import your logo (update the path as needed)
import logo from "assets/images/logo.svg";

function CustomNavbar({ ...rest }) {
  // Define the navigation links
  const navLinks = [
    { name: "Products", route: "/products" },
    { name: "Solutions", route: "/solutions" },
    { name: "Resources", route: "/resources" },
  ];

  const [mobileOpen, setMobileOpen] = useState(false);
  const [mobileView, setMobileView] = useState(false);

  // Toggle the mobile drawer
  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  // Update mobileView state based on window size
  useEffect(() => {
    const handleResize = () => {
      setMobileView(window.innerWidth < 960);
    };

    window.addEventListener("resize", handleResize);
    handleResize(); // Set initial state
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Mobile drawer content
  const drawer = (
    <Box onClick={handleDrawerToggle} sx={{ textAlign: "center" }}>
      <MKBox
        component={Link}
        to="/"
        sx={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          my: 2,
        }}
      >
        <img src={logo} alt="MLEase Logo" style={{ height: 40 }} />
      </MKBox>
      <Divider />
      <List>
        {navLinks.map((item) => (
          <ListItem key={item.name} disablePadding>
            <ListItemText
              primary={
                <MKTypography
                  component={Link}
                  to={item.route}
                  variant="button"
                  sx={{
                    textDecoration: "none",
                    color: "inherit",
                    width: "100%",
                    textAlign: "center",
                    py: 1.5,
                    fontSize: "1.1rem",
                  }}
                >
                  {item.name}
                </MKTypography>
              }
            />
          </ListItem>
        ))}
      </List>
      <Box
        sx={{
          my: 2,
          display: "flex",
          flexDirection: "column",
          gap: 1,
          px: 2,
        }}
      >
        <Button
          variant="contained"
          component={Link}
          to="/login"
          sx={{
            backgroundColor: "#FFE492",
            color: "#FD9D65",
            textTransform: "none",
            boxShadow: "none",
            fontSize: "1.1rem",
            "&:hover": {
              backgroundColor: "#FFE0A0",
            },
          }}
        >
          Login
        </Button>
        <Button
          variant="contained"
          component={Link}
          to="/get-started"
          sx={{
            backgroundColor: "#3A8DFF",
            color: "white",
            textTransform: "none",
            boxShadow: "none",
            fontSize: "1.1rem",
            "&:hover": {
              backgroundColor: "#3278d6",
            },
          }}
        >
          Get Started Free →
        </Button>
      </Box>
    </Box>
  );

  return (
    <AppBar position="sticky" sx={{ backgroundColor: "#FD9D65", boxShadow: "none" }} {...rest}>
      <Container maxWidth="lg">
        <Toolbar sx={{ justifyContent: "space-between", p: { xs: 2, md: 3 } }}>
          {/* Left: Logo */}
          <MKBox
            component={Link}
            to="/"
            sx={{
              display: "flex",
              alignItems: "center",
              textDecoration: "none",
            }}
          >
            <img src={logo} alt="MLEase Logo" style={{ height: 40 }} />
          </MKBox>

          {/* Center: Desktop Navigation */}
          {!mobileView && (
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                flexGrow: 1,
                ml: 4,
              }}
            >
              {navLinks.map((item) => (
                <MKTypography
                  key={item.name}
                  component={Link}
                  to={item.route}
                  variant="button"
                  sx={{
                    ml: 3,
                    mr: 3,
                    textDecoration: "none",
                    color: "inherit",
                    fontWeight: "bold",
                    fontSize: "1.1rem",
                    transition: "color 0.3s",
                    "&:hover": {
                      color: "#ffffff80",
                    },
                  }}
                >
                  {item.name}
                </MKTypography>
              ))}
            </Box>
          )}

          {/* Right: Action Buttons (Desktop) */}
          {!mobileView && (
            <Box sx={{ display: "flex", alignItems: "center" }}>
              <MKButton
                component={Link}
                to="/login"
                variant="contained"
                color="black"
                size="medium"
                sx={{
                  backgroundColor: "#FFE492",
                  color: "#043873",
                  textTransform: "none",
                  boxShadow: "none",
                  mr: 1,
                  fontSize: "1.1rem",
                  "&:hover": {
                    backgroundColor: "#FFE0A0",
                  },
                }}
              >
                Login
              </MKButton>
              <MKButton
                component={Link}
                to="/get-started"
                variant="contained"
                color="info"
                size="medium"
                sx={{
                  backgroundColor: "#3A8DFF",
                  color: "white",
                  textTransform: "none",
                  boxShadow: "none",
                  fontSize: "1.1rem",
                  "&:hover": {
                    backgroundColor: "#3278d6",
                  },
                }}
              >
                Get Started Free →
              </MKButton>
            </Box>
          )}

          {/* Mobile: Hamburger Menu */}
          {mobileView && (
            <Icon
              onClick={handleDrawerToggle}
              sx={{
                color: "white",
                fontSize: "2.5rem",
                cursor: "pointer",
              }}
            >
              menu
            </Icon>
          )}
        </Toolbar>
      </Container>

      {/* Mobile Drawer */}
      <Drawer
        anchor="left"
        open={mobileOpen}
        onClose={handleDrawerToggle}
        ModalProps={{ keepMounted: true }}
      >
        {drawer}
      </Drawer>
    </AppBar>
  );
}

CustomNavbar.defaultProps = {
  brand: "MLEase",
  routes: [],
};

CustomNavbar.propTypes = {
  brand: PropTypes.string,
  routes: PropTypes.array,
};

export default CustomNavbar;
