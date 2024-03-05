export const getAIMessage = async (userQuery) => {

  const message = 
    {
      role: "user",
      content: userQuery
    }

    try {
      const response = await fetch('http://localhost:8080/invoke_query', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify({ text: message })
      });

      if (!response.ok) {
          throw new Error('Failed to fetch response');
      }

      const data = await response.json();
      return {role: "assistant", content: data.message};
  } catch (error) {
      console.error('Error fetching data:', error);
      throw error;
  }
};