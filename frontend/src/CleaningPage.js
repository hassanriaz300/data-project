//This Webpage allows users to upload an Excel file containing reviews, which will be cleaned and split into separate files for further processing api.py jas endpoint for  
//preparing the reviews for classification. The user can select a file, submit it, and view the results or any errors that occur during the process.

import React, { useState } from "react";
import {
  Box,
  Typography,
  Button,
  CircularProgress,
  Alert,
  List,
  ListItem,
  Stack
} from "@mui/material";

export default function CleaningPage() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const onFileChange = e => {
    setFile(e.target.files[0]);
    setResult(null);
    setError("");
  };

  const onSubmit = async e => {
    e.preventDefault();
    if (!file) {
      setError("Please select your raw Excel file.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/prepare", {
        method: "POST",
        body: form
      });
      if (!res.ok) {
        const { detail } = await res.json().catch(() => ({}));
        throw new Error(detail || res.statusText);
      }
      const json = await res.json();
      setResult(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box p={3}>
      <Typography variant="h4" gutterBottom>
        Cleaning &amp; Splitting Reviews
      </Typography>

      <Box
        component="form"
        onSubmit={onSubmit}
        mb={3}
      >
        <Stack direction="row" alignItems="center" spacing={2}>
          <Button variant="outlined" component="label">
            Choose File
            <input
              type="file"
              hidden
              accept=".xlsx"
              onChange={onFileChange}
            />
          </Button>
          <Button
            type="submit"
            variant="contained"
            disabled={loading}
          >
            {loading
              ? <CircularProgress size={20} />
              : "Run Prepare Service"
            }
          </Button>
        </Stack>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          ⚠️ {error}
        </Alert>
      )}

      {result && (
        <Box>
          <Typography variant="h5" gutterBottom>
            Result
          </Typography>
          <Typography>
            <strong>Raw file saved to:</strong> {result.raw_file}
          </Typography>
          <Typography mt={2}>
            <strong>Generated files:</strong>
          </Typography>
          <List>
            {result.generated_files.map(fn => (
              <ListItem key={fn} disablePadding>
                • {fn}
              </ListItem>
            ))}
          </List>
        </Box>
      )}
    </Box>
  );
}
