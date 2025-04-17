/*
=========================================================
* Material Kit 2 React - v2.1.0
=========================================================
* Product Page: https://www.creative-tim.com/product/material-kit-react
* Copyright 2023 Creative Tim
* Coded by www.creative-tim.com
=========================================================
* The above copyright notice and this permission notice shall be included 
* in all copies or substantial portions of the Software.
*/

import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";

// Material Kit 2 React components
import MKBox from "components/MKBox";

// Material Kit 2 React examples
import RotatingCard from "examples/Cards/RotatingCard";
import RotatingCardFront from "examples/Cards/RotatingCard/RotatingCardFront";
import RotatingCardBack from "examples/Cards/RotatingCard/RotatingCardBack";
import DefaultInfoCard from "examples/Cards/InfoCards/DefaultInfoCard";

// Images – update these with dashboard or platform images relevant to your project
import bgFront from "assets/images/rotating-card-bg-front.jpeg";
import bgBack from "assets/images/rotating-card-bg-back.jpeg";

function Information() {
  return (
    <MKBox component="section" py={6} my={6}>
      <Container>
        <Grid container item xs={11} spacing={3} alignItems="center" sx={{ mx: "auto" }}>
          {/* Rotating Card */}
          <Grid item xs={12} lg={4} sx={{ mx: "auto" }}>
            <RotatingCard>
              <RotatingCardFront
                image={bgFront}
                icon="touch_app"
                title={
                  <>
                    Experience
                    <br />
                    MLease
                  </>
                }
                description="Streamline your ML pipeline with a unified platform designed for efficient model deployment."
              />
              <RotatingCardBack
                image={bgBack}
                title="Discover MLease"
                description="Dive into our intuitive dashboard, access real‑time analytics, and unlock end‑to‑end automation for your ML models."
                action={{
                  type: "internal",
                  route: "/features",
                  label: "Explore Features",
                }}
              />
            </RotatingCard>
          </Grid>
          {/* Info Cards */}
          <Grid item xs={12} lg={7} sx={{ ml: "auto" }}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <DefaultInfoCard
                  icon="integration_instructions"
                  title="Seamless Integration"
                  description="Connect and automate your ML workflows effortlessly with robust APIs and an intuitive dashboard."
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <DefaultInfoCard
                  icon="analytics"
                  title="Real-time Analytics"
                  description="Monitor your model performance live and make informed adjustments on the fly."
                />
              </Grid>
            </Grid>
            <Grid container spacing={3} sx={{ mt: { xs: 0, md: 6 } }}>
              <Grid item xs={12} md={6}>
                <DefaultInfoCard
                  icon="autorenew"
                  title="Automated Pipelines"
                  description="Accelerate your deployment with end‑to‑end automation that takes your models from training to production seamlessly."
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <DefaultInfoCard
                  icon="security"
                  title="Enterprise‑Grade Security"
                  description="Safeguard your data and models with advanced encryption and comprehensive security features."
                />
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </Container>
    </MKBox>
  );
}

export default Information;
