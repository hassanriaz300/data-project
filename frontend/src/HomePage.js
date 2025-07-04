import { Link as RouterLink } from "react-router-dom";
import {
  Box,
  Typography,
  Button,
  Stack
} from "@mui/material";
import ArrowRightIcon from "@mui/icons-material/ArrowForwardIos";

export default function HomePage() {
  const actions = [
    { to: "/clean", label: "Clean & Split Reviews" },
    { to: "/semantic", label: "Semantic Map (Top-N Accusations)" },
    { to: "/analysis", label: "Run Analysis" },
    { to: "/predict", label: "Quick Predict" },
    { to: "/visualize", label: "Visualize Data" },

    { to: "/deepanalyze", label: "Text & Deep Analysis" },

  ];

  return (
    <Box
      component="main"
      display="flex"
      flexDirection="column"
      alignItems="center"
      gap={4}
      py={6}
      px={2}
    >
      <Box component="header" textAlign="center" mb={2}>
        <Typography variant="h3" fontWeight="bold" gutterBottom>
          Supermarket Reviews App
        </Typography>
        <Typography variant="body1" color="text.secondary">
          What would you like to do?
        </Typography>
      </Box>

      <Stack spacing={2} width="100%" maxWidth={400}>
        {actions.map(({ to, label }) => (
          <Button
            key={to}
            component={RouterLink}
            to={to}
            variant="contained"
            fullWidth
            endIcon={<ArrowRightIcon />}
            sx={{ justifyContent: "space-between", py: 1.5 }}
          >
            {label}
          </Button>
        ))}
      </Stack>
    </Box>
  );
}
