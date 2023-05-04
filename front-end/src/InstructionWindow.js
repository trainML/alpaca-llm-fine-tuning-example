import React from 'react';

import {
  TextField,
  Grid,
  Stack,
  Typography,
  Paper,
  CircularProgress,
  Box,
  Container,
} from '@mui/material';
import { LoadingButton } from '@mui/lab';

function InstructionWindow({
  instruction,
  setInstruction,
  input,
  setInput,
  response,
  loading,
  disabled,
  send,
}) {
  const handleInstructionChange = (event) => {
    const { value: newValue } = event.target;
    setInstruction(newValue);
  };
  const handleInputChange = (event) => {
    const { value: newValue } = event.target;
    setInput(newValue);
  };

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6} lg={6}>
        <Stack spacing={2} direction='column'>
          <TextField
            sx={{ width: '100%' }}
            id='instruction'
            label='Instruction'
            multiline
            rows={2}
            value={instruction}
            onChange={handleInstructionChange}
            required
          />
          <TextField
            sx={{ width: '100%' }}
            id='input'
            label='Input'
            multiline
            rows={10}
            value={input}
            onChange={handleInputChange}
          />
          <Stack
            spacing={2}
            direction='row'
            justifyContent='center'
            alignItems='center'
          >
            <LoadingButton
              variant='contained'
              color='brand'
              onClick={send}
              loading={loading}
              disabled={disabled}
            >
              Send
            </LoadingButton>
          </Stack>
        </Stack>
      </Grid>
      <Grid item xs={12} md={6} lg={6}>
        <Paper
          variant='outlined'
          style={{
            display: 'flex',
            width: '100%',
            height: '100%',
            flexGrow: 1,
          }}
        >
          {loading ? (
            <Box
              style={{
                display: 'flex',
                width: '100%',
                height: '100%',
                flexGrow: 1,
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <CircularProgress />
            </Box>
          ) : response ? (
            <Container sx={{ paddingY: '15px' }}>
              <Typography variant='body' sx={{ whiteSpace: 'pre-wrap' }}>
                {response}
              </Typography>
            </Container>
          ) : undefined}
        </Paper>
      </Grid>
    </Grid>
  );
}

export default InstructionWindow;
