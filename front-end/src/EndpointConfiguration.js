import React, { useState } from 'react';

import {
  Button,
  Box,
  Typography,
  Grid,
  Toolbar,
  Stack,
  FormControl,
  TextField,
  FormHelperText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  AccordionActions,
} from '@mui/material';

import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

function EndpointConfiguration({ current_values, setConfiguration }) {
  const [expanded, setExpanded] = useState(true);
  const [pristine, setPristine] = useState(true);
  const [endpoint, setEndpoint] = useState(
    current_values ? current_values.endpoint : ''
  );
  const [max_tokens, setMaxTokens] = useState(
    current_values ? current_values.max_tokens : 500
  );
  const [temperature, setTemperature] = useState(
    current_values ? current_values.temperature : 1.0
  );
  const [top_p, setTopP] = useState(
    current_values ? current_values.top_p : 1.0
  );
  const [num_beams, setNumBeams] = useState(
    current_values ? current_values.num_beams : 1
  );

  const handleSubmit = () => {
    setConfiguration({
      endpoint,
      max_tokens,
      temperature,
      top_p,
      num_beams,
    });
    setExpanded(false);
  };
  return (
    <Accordion expanded={expanded} onChange={() => setExpanded(!expanded)}>
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        aria-controls='panel1a-content'
        id='panel1a-header'
      >
        <Typography variant='h6'>Configuration</Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Grid container spacing={2}>
          <Grid item xs={12} md={12} lg={12}>
            <Box>
              <Grid container spacing={2}>
                <Grid item xs={12} md={12} lg={12}>
                  <Typography variant='subtitle'>Endpoint</Typography>
                </Grid>
                <Grid item xs={12} md={12} lg={12}>
                  <Box>
                    <FormControl fullWidth required>
                      <TextField
                        label='Endpoint Address'
                        id='endpoint'
                        aria-describedby='endpoint-helper-text'
                        variant='outlined'
                        onChange={(event) => {
                          setPristine(false);
                          setEndpoint(event.target.value);
                        }}
                        value={endpoint}
                      />

                      <FormHelperText id='endpoint-helper-text'>
                        Enter the LLM endpoint URL
                      </FormHelperText>
                    </FormControl>
                  </Box>
                </Grid>
              </Grid>
            </Box>
          </Grid>

          <Grid item xs={12} md={12} lg={12}>
            <Box>
              <Grid container spacing={2}>
                <Grid item xs={12} md={12} lg={12}>
                  <Typography variant='subtitle'>
                    Generation Settings
                  </Typography>
                </Grid>
                <Grid item xs={12} md={12} lg={12}>
                  <Box>
                    <FormControl fullWidth>
                      <TextField
                        label='Max Tokens'
                        id='max_tokens'
                        type='number'
                        aria-describedby='max-tokens-helper-text'
                        variant='outlined'
                        onChange={(event) => {
                          setPristine(false);
                          setMaxTokens(event.target.value);
                        }}
                        value={max_tokens}
                      />

                      <FormHelperText id='max-tokens-helper-text'>
                        Enter the maximum tokens to generate (includes input
                        tokens)
                      </FormHelperText>
                    </FormControl>
                  </Box>
                </Grid>
                <Grid item xs={12} md={12} lg={12}>
                  <Box>
                    <FormControl fullWidth>
                      <TextField
                        label='Temperature'
                        id='temperature'
                        type='number'
                        aria-describedby='temperature-helper-text'
                        variant='outlined'
                        onChange={(event) => {
                          setPristine(false);
                          setTemperature(event.target.value);
                        }}
                        value={temperature}
                      />

                      <FormHelperText id='temperature-helper-text'>
                        Enter the sampling temperature to use, between 0 and 2.
                      </FormHelperText>
                    </FormControl>
                  </Box>
                </Grid>
                <Grid item xs={12} md={12} lg={12}>
                  <Box>
                    <FormControl fullWidth>
                      <TextField
                        label='Top P'
                        id='top_p'
                        type='number'
                        aria-describedby='top-p-helper-text'
                        variant='outlined'
                        onChange={(event) => {
                          setPristine(false);
                          setTopP(event.target.value);
                        }}
                        value={top_p}
                      />

                      <FormHelperText id='top-p-helper-text'>
                        Enter the top_p probability for nucleus sampling,
                        between 0 and 1.
                      </FormHelperText>
                    </FormControl>
                  </Box>
                </Grid>

                <Grid item xs={12} md={12} lg={12}>
                  <Box>
                    <FormControl fullWidth>
                      <TextField
                        sx={{ width: '100%' }}
                        id='num_beams'
                        label='Beams'
                        value={num_beams}
                        aria-describedby='stop-helper-text'
                        onChange={(event) => {
                          setPristine(false);
                          setNumBeams(event.target.value);
                        }}
                      />
                      <FormHelperText id='stop-helper-text'>
                        Number of beams for beam search. 1 means no beam search.
                      </FormHelperText>
                    </FormControl>
                  </Box>
                </Grid>
              </Grid>
            </Box>
          </Grid>
        </Grid>
      </AccordionDetails>
      <AccordionActions>
        <Toolbar>
          <Stack spacing={2} direction='row'>
            <Button
              variant='contained'
              color='brand'
              onClick={handleSubmit}
              disabled={pristine || endpoint === ''}
            >
              Update
            </Button>

            <Button
              variant='outlined'
              onClick={() => {
                setEndpoint(current_values ? current_values.endpoint : '');
                setMaxTokens(current_values ? current_values.max_tokens : 500);
                setTemperature(
                  current_values ? current_values.temperature : 1.0
                );
                setTopP(current_values ? current_values.top_p : 1.0);
                setNumBeams(current_values ? current_values.num_beams : 1);
              }}
            >
              Cancel
            </Button>
          </Stack>
        </Toolbar>
      </AccordionActions>
    </Accordion>
  );
}

export default EndpointConfiguration;
