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

export default function SemanticPage() {
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
      setError("Select your *cleaned-wide* Excel first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/semantic/map", { method: "POST", body: form });

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
        Semantic Accusation Mapping
      </Typography>

      <Box component="form" onSubmit={onSubmit} mb={3}>
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
          <Button type="submit" variant="contained" disabled={loading}>
            {loading ? <CircularProgress size={20} /> : "Run Semantic Service"}
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
