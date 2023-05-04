import { useState } from 'react';
import axios from 'axios';
import { Grid, Container } from '@mui/material';

import Header from './Header';
import EndpointConfiguration from './EndpointConfiguration';

import ErrorSnackBar from './ErrorSnackBar';
import InstructionWindow from './InstructionWindow';

function App() {
  const [api_error, setApiError] = useState('');
  const [instruction, setInstruction] = useState('');
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');
  const [endpoint, setEndpoint] = useState('');
  const [max_tokens, setMaxTokens] = useState(2048);
  const [temperature, setTemperature] = useState(0.7);
  const [top_p, setTopP] = useState(0.2);
  const [num_beams, setNumBeams] = useState(1);
  const [loading, setLoading] = useState(false);

  const setConfiguration = (values) => {
    console.log(values);
    setEndpoint(values.endpoint);
    setMaxTokens(values.max_tokens);
    setTemperature(values.temperature);
    setTopP(values.top_p);
    setNumBeams(values.num_beams);
    setInstruction('');
    setResponse('');
  };

  const sendInstruction = async () => {
    setLoading(true);
    let payload;

    payload = {
      instruction: instruction,
      input: Boolean(input) ? input : undefined,
      max_tokens,
      temperature,
      top_p,
      num_beams,
    };
    setResponse('');
    try {
      const result = await axios.post(`${endpoint}/instruct`, payload);

      setResponse(result.data);
    } catch (error) {
      console.log(error);
      if (error.response) {
        setApiError(
          `An error occured: ${error.response.status} - ${JSON.stringify(
            error.response.data
          )} `
        );
      } else {
        setApiError(
          'Network Error Occurred.  Either endpoint URL is invalid or generation exceeded 100s.'
        );
      }
    }
    setLoading(false);
  };

  return (
    <Grid>
      <Header />
      <br />
      <Container>
        <Grid container xs={12} spacing={2}>
          <Grid item xs={12}>
            <InstructionWindow
              instruction={instruction}
              setInstruction={setInstruction}
              response={response}
              input={input}
              setInput={setInput}
              send={sendInstruction}
              loading={loading}
              disabled={!Boolean(endpoint) || !Boolean(instruction)}
            />
          </Grid>
          <Grid item xs={12}>
            <EndpointConfiguration
              current_values={{
                endpoint,
                max_tokens,
                temperature,
                top_p,
                num_beams,
              }}
              setConfiguration={setConfiguration}
            />
          </Grid>
        </Grid>
        <ErrorSnackBar
          message={api_error}
          clearMessage={() => setApiError('')}
        />
      </Container>
    </Grid>
  );
}

export default App;
